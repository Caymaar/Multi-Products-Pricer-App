import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Sequence, Union
from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreeModel, TreePortfolio
from pricers.bs_pricer import BSPortfolio

from risk_metrics.rate_product_sensitivity import dv01, duration, convexity
from risk_metrics.greeks import GreeksCalculator

from rate.product import (
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    InterestRateSwap
)


class SensitivityAnalyzer:
    """
    Calcule Delta, Gamma, Vega, Theta, Rho, Speed pour une stratégie/portefeuille
    en s'appuyant soit sur GreeksCalculator (MC/Tree) soit sur BSPortfolio (BS).
    Permet aussi de tracer l'évolution du prix en fonction du spot, de la vol,
    de la maturité ou d'un shift parallèle de taux.
    """

    def __init__(self,
                 strategy: Any,
                 engine: Any,
                 epsilon: float = 1e-4,
                 method: str | None = None):
        """
        :param strategy: instance de Strategy (implémente price(engine), get_legs()).
        :param engine:    MonteCarloEngine, TreePortfolio/TreeModel ou BSPortfolio.
        :param epsilon:   bump pour dérivées finies.
        :param method:    type de pricing MC ("MC","Longstaff") ou None pour auto.
        """
        self.strategy = strategy
        self.engine = engine
        self.epsilon = epsilon
        self.method = method or self._infer_method()
        # Choix du calculateur de greeks
        if isinstance(engine, (MonteCarloEngine, TreeModel, TreePortfolio)):
            self._calc = GreeksCalculator(engine, epsilon=epsilon, type=self.method)
        elif isinstance(engine, BSPortfolio):
            self._calc = engine
        else:
            raise ValueError(f"Engine non supporté : {type(engine)}")

    def _infer_method(self) -> str:
        if isinstance(self.engine, MonteCarloEngine):
            return "MC"
        if isinstance(self.engine, (TreeModel, TreePortfolio)):
            return "Trinomial"
        return ""

    def delta(self) -> float:
        """Delta de la stratégie / portefeuille."""
        return self._calc.delta()

    def gamma(self) -> float:
        """Gamma de la stratégie / portefeuille."""
        return self._calc.gamma()

    def vega(self) -> float:
        """Vega de la stratégie / portefeuille."""
        return self._calc.vega()

    def theta(self) -> float:
        """Theta de la stratégie / portefeuille."""
        return self._calc.theta()

    def rho(self) -> float:
        """Rho de la stratégie / portefeuille."""
        return self._calc.rho()


    def underlying_sensitivity(self,
                                S_range: Sequence[float]) -> np.ndarray:
        """
        Prix du portefeuille pour chaque S0 dans S_range.
        """
        vals = []
        for S in S_range:
            # bump spot
            mkt = self.engine.market.copy(S0=S)
            eng = self.engine.recreate_model(market=mkt) if hasattr(self.engine, "recreate_model") else self._deep_copy_engine(mkt)
            vals.append(self.strategy.price(eng))
        return np.array(vals)

    def volatility_sensitivity(self,
                                vol_range: Sequence[float]) -> np.ndarray:
        """
        Prix du portefeuille pour chaque vol dans vol_range.
        """
        vals = []
        for sig in vol_range:
            mkt = self.engine.market.copy(sigma=sig)
            eng = self.engine.recreate_model(market=mkt) if hasattr(self.engine, "recreate_model") else self._deep_copy_engine(mkt)
            vals.append(self.strategy.price(eng))
        return np.array(vals)

    def maturity_sensitivity(self,
                              maturities: Sequence[Any]) -> np.ndarray:
        """
        Prix du portefeuille pour chaque date de maturité dans maturities.
        """
        vals = []
        for T in maturities:
            # bump maturity sur la stratégie
            #self.strategy.maturity_date = T
            for idx in range(len(self.strategy.options)):
                self.strategy.options[idx].maturity_date = T
            vals.append(self.strategy.price(self.engine))
        return np.array(vals)

    def rate_shift_sensitivity(self,
                               shifts: Sequence[float]) -> np.ndarray:
        """
        Prix du portefeuille pour un shift parallèle des taux discount.
        :param shifts: liste de h en décimal (e.g. ±0.001 = ±10bp)
        """
        vals = []
        base = self.engine.market
        for h in shifts:
            def df_shifted(t: float) -> float:
                return base.discount(t) * np.exp(-h * t)
            def zero_rate_shifted(t: float) -> float:
                return base.zero_rate(t) + h
            mkt = base.copy(discount_curve=df_shifted, zero_rate_curve=zero_rate_shifted)
            eng = self.engine.recreate_model(market=mkt) if hasattr(self.engine, "recreate_model") else self._deep_copy_engine(mkt)
            vals.append(self.strategy.price(eng))
        return np.array(vals)

    def plot(self,
             x: Sequence,
             y: Sequence,
             xlabel: str,
             ylabel: str,
             title: str) -> None:
        """Trace y vs x."""
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, lw=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, ls=":")
        plt.show()

    def _deep_copy_engine(self, market: Any) -> Any:
        """Fallback si pas de recreate_model."""
        import copy
        eng = copy.deepcopy(self.engine)
        eng.market = market
        return eng

    def _rate_legs(self):
        """
        Renvoie la liste des (leg, weight) qui sont des produits de taux.
        """
        return [
            (leg, w)
            for leg, w in self.strategy.get_legs()
            if isinstance(leg, (ZeroCouponBond,
                                FixedRateBond,
                                FloatingRateBond,
                                InterestRateSwap))
        ]

    def rate_greeks(self) -> dict:
        """
        Agrège les sensibilités de taux (DV01, Duration, Convexity)
        sur tous les legs de taux de la stratégie.
        """
        rate_legs = self._rate_legs()
        if not rate_legs:
            return {}
        df = self.engine.market.discount  # callable DF(t)
        return {
            'dv01':      sum(w * dv01(leg, df)      for leg, w in rate_legs),
            'duration':  sum(w * duration(leg, df)  for leg, w in rate_legs),
            'convexity': sum(w * convexity(leg, df) for leg, w in rate_legs),
        }

    def plot_rate_sensitivity(self,
                              shifts:   Union[np.ndarray, list[float]],
                              xlabel:   str = "Shift taux (décimal)",
                              ylabel:   str = "Prix strat.",
                              title:    str = "Sensibilité parallèle taux"):
        """
        Trace le prix de la stratégie vs shift parallèle des taux,
        et la droite de pente DV01 (approx. linéaire).
        """
        # 1) prix exacts sous chaque shift
        prices = self.rate_shift_sensitivity(shifts)
        # 2) pente DV01 agrégée
        rk     = self.rate_greeks().get('dv01', 0.0)
        # 3) approximation linéaire : P0 + dv01 * shift
        P0     = self.strategy.price(self.engine)
        approx = P0 + np.array(shifts) * rk

        # 4) tracé
        plt.figure(figsize=(8, 4))
        plt.plot(shifts, prices, lw=2, label="Prix exact")
        plt.plot(shifts, approx, ls="--", label="Linearisation")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, ls=":")
        plt.show()

    def all_greeks(self) -> dict:
        """
        Retourne un dict complet :
          - Delta, Gamma, Vega, Theta, Rho (options)
          - DV01, Duration, Convexity (taux)
        """
        res = {
            'delta':   self.delta(),
            'gamma':   self.gamma(),
            'vega':    self.vega(),
            'theta':   self.theta(),
            'rho':     self.rho(),
        }
        res.update(self.rate_greeks())
        return res


if __name__ == "__main__":
    from datetime import datetime, timedelta
    import numpy as np

    from market.market_factory import create_market
    from option.option import OptionPortfolio
    from investment_strategies.vanilla_strategies import (
        BearCallSpread, BullCallSpread, ButterflySpread, Straddle
    )
    from pricers.mc_pricer import MonteCarloEngine

    # 1) Construction du marché au 25/04/2023
    pricing_date = datetime(2023, 4, 25)
    maturity_date = pricing_date + timedelta(days=365)
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    market = create_market(
        stock="LVMH",
        pricing_date=pricing_date,
        vol_source="implied",
        hist_window=252,
        curve_method="svensson",
        curve_kwargs={"initial_guess": sv_guess},
        dcc="Actual/365"
    )

    S0 = market.S0
    sigma = market.sigma

    # 2) Choix de la stratégie (ex : Bull Call Spread)
    #    on définit strikes en % de S0
    k_buy = 0.95 * S0
    k_sell = 1.05 * S0
    strat = BullCallSpread(
        strike_buy=k_buy,
        strike_sell=k_sell,
        pricing_date=pricing_date,
        maturity_date=maturity_date
    )

    # 3) Préparation du portefeuille d’options et de l’engine MC
    legs, weights = zip(*strat.get_legs())
    ptf = OptionPortfolio(list(legs), list(weights))

    mc = MonteCarloEngine(
        market=market,
        option_ptf=ptf,
        pricing_date=pricing_date,
        n_paths=50_000,
        n_steps=200,
        seed=42,
        compute_antithetic=True
    )

    bse = BSPortfolio(
        market=market,
        option_ptf=ptf,
        pricing_date=pricing_date
    )

    te = TreePortfolio(
        market=market,
        option_ptf=ptf,
        pricing_date=pricing_date,
        n_steps=300)


    # 4) Instanciation de l’analyseur de sensibilité
    an = SensitivityAnalyzer(strategy=strat, engine=mc, epsilon=1e-4)

    # 5) Calcul et affichage des greeks
    print("Delta :", an.delta())
    print("Gamma :", an.gamma())
    print("Vega  :", an.vega())
    print("Theta :", an.theta())
    print("Rho   :", an.rho())

    # 6) Sensibilité au Spot (80%→120% S0)
    S_range = np.linspace(0.8 * S0, 1.2 * S0, 25)
    vals_spot = an.underlying_sensitivity(S_range)
    an.plot(
        x=S_range,
        y=vals_spot,
        xlabel="Prix du sous-jacent S₀",
        ylabel="Valeur stratégie",
        title="Sensibilité au Spot"
    )

    # 7) Sensibilité à la Volatilité (50%→150% σ)
    vol_range = np.linspace(0.5 * sigma, 1.5 * sigma, 25)
    vals_vol = an.volatility_sensitivity(vol_range)
    an.plot(
        x=vol_range,
        y=vals_vol,
        xlabel="Volatilité σ",
        ylabel="Valeur stratégie",
        title="Sensibilité à la Volatilité"
    )

    # 8) Sensibilité aux shifts de taux parallèles (±100 bp)
    shifts = np.linspace(-0.01, 0.01, 25)
    vals_rate = an.rate_shift_sensitivity(shifts)
    an.plot(
        x=shifts * 1e4,  # afficher en bp
        y=vals_rate,
        xlabel="Shift taux (bp)",
        ylabel="Valeur stratégie",
        title="Sensibilité aux taux"
    )

    # 9) Sensibilité à la maturité (0.5→1.5 an(s))
    mat_range = [
        pricing_date + timedelta(days=int(365 * t))
        for t in np.linspace(0.5, 1.5, 7)
    ]
    vals_mat = an.maturity_sensitivity(mat_range)
    an.plot(
        x=[(T - pricing_date).days / 365 for T in mat_range],
        y=vals_mat,
        xlabel="Maturité (années)",
        ylabel="Valeur stratégie",
        title="Sensibilité à la maturité"
    )
