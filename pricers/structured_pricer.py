import numpy as np
from typing import Tuple, List
from datetime import datetime
from rate.curve_utils import make_zc_curve
from rate.products import ZeroCouponBond
from option.option import Option, OptionPortfolio
from market.market import Market
from stochastic_process.gbm_process import GBMProcess
from pricers.mc_pricer import MonteCarloEngine


class StructuredPricer:
    def __init__(self,
                 market: Market,
                 pricing_date: datetime,
                 zc_method: str,
                 zc_args: tuple,
                 n_paths: int = 100_000,
                 n_steps: int = 300,
                 seed: int = None,
                 compute_antithetic: bool = False):
        self.market = market
        self.pricing_date = pricing_date
        self.zc_curve = make_zc_curve(zc_method, *zc_args)
        self.dcc = market.DaysCountConvention
        # paramètres pour simuler le sous-jacent
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.compute_antithetic = compute_antithetic

    def get_mc_engine(self, opt: Option) -> MonteCarloEngine:
        """
        Usine à MonteCarloEngine pour pricer une seule Option.
        """
        return MonteCarloEngine(
            market=self.market,
            option_ptf=OptionPortfolio([opt]),
            pricing_date=self.pricing_date,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            seed=self.seed,
            compute_antithetic=self.compute_antithetic
        )

    def price_zcb(self, zcb: ZeroCouponBond) -> float:
        return zcb.price(self.zc_curve)

    def simulate_underlying(self,
                            maturity_date: datetime,
                            obs_dates: List[datetime]
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simule un GBM du sous-jacent :
         - S     : array (n_paths, n_steps+1)
         - times : array des year-fractions pour chaque obs_date
        """
        T = self.dcc.year_fraction(self.pricing_date, maturity_date)
        dt = T / self.n_steps
        t_div = None
        if self.market.div_date is not None:
            T_div = self.dcc.year_fraction(self.pricing_date, self.market.div_date)
            t_div = int(T_div / dt)
        gbm = GBMProcess(
            market=self.market,
            dt=dt,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            t_div=t_div,
            compute_antithetic=self.compute_antithetic,
            seed=self.seed
        )
        S = gbm.simulate()
        times = np.array([
            self.dcc.year_fraction(self.pricing_date, d)
            for d in obs_dates
        ])
        return S, times

    def compute_autocall_payoffs(self,
                                 S: np.ndarray,
                                 coupon_barrier:     float,
                                 call_barrier:       float,
                                 protection_barrier: float,
                                 coupon_rates:       np.ndarray,
                                 obs_dates:          List[datetime],
                                 notional:           float
                                ) -> Tuple[np.ndarray, np.ndarray]:
        n_paths, n_steps = S.shape
        S0 = self.market.S0
        payoffs = np.zeros(n_paths)
        red_times = np.zeros(n_paths)
        times = np.array([self.dcc.year_fraction(self.pricing_date, d) for d in obs_dates])
        accruals = np.diff(np.concatenate([[0.0], times]))
        coupon_amts = coupon_rates * notional * accruals
        idxs = (times / times[-1] * (n_steps-1)).round().astype(int)

        # observations intermédiaires
        for i, idx in enumerate(idxs):
            alive = payoffs == 0
            Si = S[alive, idx]
            ids = np.where(alive)[0]
            mask_call = Si >= call_barrier * S0
            payoffs[ids[mask_call]]  = coupon_amts[i] + notional
            red_times[ids[mask_call]] = times[i]
            mask_coup = (Si >= coupon_barrier * S0) & (~mask_call)
            payoffs[ids[mask_coup]] = coupon_amts[i]

        # survivants à maturité
        surv = payoffs == 0
        ST = S[surv, -1]
        mask_prot = ST >= protection_barrier * S0
        payoffs[surv][mask_prot] = coupon_amts[-1] + notional
        payoffs[surv][~mask_prot] = (ST[~mask_prot] / S0) * notional
        red_times[surv] = times[-1]

        return payoffs, red_times

    def discount(self,
                 cashflows: np.ndarray,
                 times:     np.ndarray
                ) -> np.ndarray:
        dfs = np.array([self.zc_curve(t) for t in times])
        return cashflows * dfs