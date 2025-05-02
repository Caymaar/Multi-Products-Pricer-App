import numpy as np
from typing import Tuple, List
from datetime import datetime
from market.day_count_convention import DayCountConvention
from rate.products import ZeroCouponBond
from option.option import Option, OptionPortfolio
from market.market import Market
from stochastic_process.gbm_process import GBMProcess
from pricers.mc_pricer import MonteCarloEngine


class StructuredPricer:
    def __init__(self,
                 market: Market,
                 pricing_date: datetime,
                 df_curve,
                 maturity_date: datetime,
                 n_paths: int = 100_000,
                 n_steps: int = 300,
                 seed: int = None,
                 compute_antithetic: bool = False):
        self.market = market
        self.pricing_date = pricing_date
        self.df_curve = df_curve
        self.maturity_date = maturity_date
        self.dcc = market.dcc
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
        return zcb.price(self.df_curve)

    def simulate_underlying(self,
                            frequency: str,
                            dcc: DayCountConvention
                           ) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        Simule un GBM du sous-jacent :
         - S     : array (n_paths, n_steps+1)
         - times : array des year-fractions pour chaque obs_date
        """
        T = dcc.year_fraction(self.pricing_date, self.maturity_date)
        dt = T / self.n_steps
        t_div = None
        if self.market.div_date is not None:
            T_div = dcc.year_fraction(self.pricing_date, self.market.div_date)
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

        obs_dates = dcc.schedule(self.pricing_date, self.maturity_date, frequency)
        times     = np.array([
            self.dcc.year_fraction(self.pricing_date, d)
            for d in obs_dates
        ])

        return S, times, obs_dates

    def compute_autocall_cashflows(self,
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
        # times & accruals
        times = np.array([self.dcc.year_fraction(self.pricing_date, d) for d in obs_dates])
        accruals = np.diff(np.concatenate([[0.0], times]))
        coupon_amts = coupon_rates * notional * accruals
        idxs = (times / times[-1] * (n_steps-1)).round().astype(int)

        # cashflow matrix : one column per obs + one for maturity
        n_obs = len(obs_dates)
        cashflows = np.zeros((n_paths, n_obs))
        called = np.zeros(n_paths, dtype=bool)

        # intermediate observations
        for i, idx in enumerate(idxs):
            alive_ids = np.where(~called)[0]
            Si = S[alive_ids, idx]

            # first, any auto‐call
            mask_call = Si >= call_barrier * S0
            idx_call = alive_ids[mask_call]
            cashflows[idx_call, i] = coupon_amts[i] + notional
            called[idx_call] = True

            # then, coupons for those still alive but above coupon barrier
            alive2 = np.where(~called)[0]
            Si2 = S[alive2, idx]
            mask_coup = Si2 >= coupon_barrier * S0
            idx_coup = alive2[mask_coup]
            cashflows[idx_coup, i] = coupon_amts[i]

        # maturity cashflows for survivors
        surv_ids = np.where(~called)[0]
        ST = S[surv_ids, -1]
        mask_prot = ST >= protection_barrier * S0
        idx_prot  = surv_ids[mask_prot]
        idx_loss  = surv_ids[~mask_prot]
        cashflows[idx_prot, -1] += notional
        cashflows[idx_loss, -1] += (ST[~mask_prot] / S0) * notional

        # return CF matrix and times including maturity
        return cashflows, times

    def discount_cf(self,
                 cashflows: np.ndarray,
                 times:     np.ndarray
                ) -> np.ndarray:
        dfs = np.array([self.df_curve(t) for t in times])
        return cashflows * dfs