from rate.interpolation import RateInterpolation
from rate.vasicek import VasicekModel
from rate.nelson_siegel import NelsonSiegelModel
from rate.svensson import SvenssonModel
import numpy as np
from datetime import datetime
import pandas as pd
from data.management.data_utils import tenor_to_years

class ZeroCouponCurveBuilder:
    def __init__(self, taus, swap_rates, freq=1):
        self.taus = np.array(taus)
        self.swap_rates = np.array(swap_rates)
        self.freq = freq
        self.dfs = None
        self.zero_rates = None
        self._bootstrap()

    def _bootstrap(self):
        year_fracs = np.diff(np.concatenate(([0.0], self.taus)))
        dfs = np.zeros(len(self.taus))
        for i, tau in enumerate(self.taus):
            c = self.swap_rates[i]
            if i == 0:
                dfs[i] = 1 / (1 + c * year_fracs[i])
            else:
                fixed_pv = np.sum(self.swap_rates[:i] * year_fracs[:i] * dfs[:i])
                dfs[i] = (1 - fixed_pv) / (1 + c * year_fracs[i])
        self.dfs = dfs
        self.zero_rates = -np.log(dfs) / self.taus

    def build_curve(self,
                    method: str = 'interpolation',
                    **kwargs):
        """
        Construit la courbe zéro-coupon selon la méthode spécifiée.

        :param method: 'interpolation', 'nelson-siegel', 'svensson', 'vasicek'
        :param kind: type d'interpolation ('linear', 'cubic', 'refined cubic')
        :param initial_guess: pour modèles paramétriques
        :param dt: pas de temps pour Vasicek
        :param n_steps: nombre de pas pour Vasicek
        :return: instance de courbe (InterpolatedZCCurve ou ParametricZCCurve)
        """
        method = method.lower()
        if method == 'interpolation':
            kind = kwargs.get('kind', 'linear')
            interp = RateInterpolation(self.taus, self.zero_rates, kind=kind)
            return interp

        elif method == 'nelson-siegel':
            initial_guess = kwargs.get('initial_guess')
            if initial_guess is None:
                raise ValueError("initial_guess requis pour Nelson-Siegel")
            model = NelsonSiegelModel.calibrate(self.taus, self.zero_rates, initial_guess)
            return model

        elif method == 'svensson':
            initial_guess = kwargs.get('initial_guess')
            if initial_guess is None:
                raise ValueError("initial_guess requis pour Svensson")
            model = SvenssonModel.calibrate(self.taus, self.zero_rates, initial_guess)
            return model

        elif method == 'vasicek':
            dt = kwargs.get('dt')
            n_steps = kwargs.get('n_steps')
            if dt is None or n_steps is None:
                raise ValueError("dt et n_steps requis pour Vasicek")
            model = VasicekModel.calibrate(self.zero_rates, dt, n_steps)
            return model

        else:
            raise ValueError(f"Méthode inconnue : {method}")


class ZCFactory:
    def __init__(self, risk_free_curve: pd.Series, floating_curve: pd.Series, dcc="Actual/365"):
        self.risk_free_curve = risk_free_curve
        self.floating_curve = floating_curve
        self.dcc = dcc

    def get_maturity_and_rates(self, curve_type: str = 'discount', dcc="Actual/365"):
        curve_df = self.risk_free_curve if curve_type == 'discount' else self.floating_curve 
        maturities = np.array([tenor_to_years(tenor=t, dcc=dcc) for t in curve_df.index])
        rates = curve_df.values
        return maturities, rates

    def make_zc_curve(self,
             method: str = "interpolation",
             curve_type: str = "discount",
             dcc:str = "Actual/365",
             **kwargs):
        """
        Construit une courbe ZC selon la méthode spécifiée.

        :param method: 'interpolation', 'nelson-siegel', 'svensson', 'vasicek'
        :param curve_type: 'discount', 'forward'
        :param **kwargs : ->
                        initial_guess: paramètres initiaux de calibration (modèles paramétriques)
                        pas de temps (modèles stochastiques)
                        n_steps: nombre d'étapes (modèles stochastiques)
        :return: instance d'une courbe implémentant AbstractYieldCurve
        """
        method = method.lower()
        maturities, rates = self.get_maturity_and_rates(curve_type, dcc=dcc)
        builder = ZeroCouponCurveBuilder(maturities, rates)
        return builder.build_curve(method=method,**kwargs)

    def discount_curve(self, method: str = "interpolation", **kwargs):
        """
        Renvoie une fonction DF(t) pour actualiser des flux, à partir de la courbe ZC calibrée.

        :param method: méthode de calibration (interpolation, nelson-siegel, etc.)
        :param kwargs: paramètres du modèle
        :return: fonction DF(t) = exp(-r(t) * t)
        """
        zc_curve = self.make_zc_curve(method=method, curve_type='discount', **kwargs)

        def discount_factor(t: float) -> float:
            r = zc_curve.yield_value(t)
            return np.exp(-r * t)

        return discount_factor

    def forward_curve(self, method: str = "interpolation", **kwargs):
        """
        Renvoie une fonction f(t1, t2) donnant le taux forward implicite projeté (ex. Euribor 3M).

        :param date: date de valorisation
        :param method: méthode de calibration (interpolation, nelson-siegel, etc.)
        :param kwargs: paramètres du modèle
        :return: fonction f(t1, t2)
        """
        zc_curve = self.make_zc_curve(method=method, curve_type='forward', **kwargs)

        return zc_curve.forward_rate

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from data.management.data_retriever import DataRetriever
    from data.management.data_utils import tenor_to_years

    # === Paramètres généraux ===
    source = "LVMH"
    valuation_date = datetime(year=2023, month=10, day=1)
    dcc = "Actual/365"

    data = DataRetriever(source)
    rf_curve = data.get_risk_free_curve(valuation_date)
    fl_curve = data.get_floating_curve(valuation_date)
    zcf = ZCFactory(risk_free_curve=rf_curve, floating_curve=fl_curve, dcc=dcc)
    tenors = np.array([tenor_to_years(tenor=t, dcc=dcc) for t in rf_curve.index])


    # === Paramètres initiaux pour modèles ===
    ns_guess = [0.02, -0.01, 0.01, 1.5]
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    # === Grille temporelle ===
    T = np.linspace(min(tenors), max(tenors), 200)
    T1 = T
    T2 = T + 0.25  # pour calcul forward 3M

    # === Construction des courbes ===
    methods = {
        "Interpolation (refined cubic)": zcf.make_zc_curve(date=valuation_date, method="interpolation", kind="refined cubic"),
        "Nelson-Siegel": zcf.make_zc_curve(date=valuation_date, method="nelson-siegel", initial_guess=ns_guess),
        "Svensson": zcf.make_zc_curve(date=valuation_date, method="svensson", initial_guess=sv_guess),
    }

    # === TRAÇAGE ZC ===
    plt.figure(figsize=(10, 6))
    for label, zc_model in methods.items():
        zc_yields = zc_model.yield_curve_array(T)
        plt.plot(T, zc_yields, label=label)
    plt.scatter(tenors, rf_curve, color="black", label="Taux observés", zorder=5)
    plt.title("Courbes Zéro-Coupon calibrées (Interpolation / NS / Svensson)")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux Zéro-Coupon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === TRAÇAGE FORWARD ===
    plt.figure(figsize=(10, 6))
    for label, zc_model in methods.items():
        fwd_rates = [zc_model.forward_rate(t1, t2) for t1, t2 in zip(T1, T2)]
        plt.plot(T1, fwd_rates, label=f"{label} - Forward 3M")
    plt.title("Courbes Forward implicites (3M)")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux Forward 3M")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()