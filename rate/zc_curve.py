from rate.interpolation import RateInterpolation
from rate.vasicek import VasicekModel
from rate.nelson_siegel import NelsonSiegelModel
from rate.svensson import SvenssonModel
import numpy as np
from datetime import datetime

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
            model = VasicekModel.calibrate(self.zero_rates, self.dt, n_steps)
            return model

        else:
            raise ValueError(f"Méthode inconnue : {method}")


class ZCFactory:
    def __init__(self, source: str = "AMAZON", date: datetime = datetime(year=2023,month=10,day=1)):
        self.retriever = DataRetriever(source)
        self.date = date

    def get_maturity_and_rates(self, date: datetime):
        curve_df = self.retriever.get_risk_free_curve(date) / 100
        maturities = np.array([tenor_to_years(t) for t in curve_df.index])
        rates = curve_df.values
        return maturities, rates

    def make_zc_curve(self,
             date: datetime,
             method: str = "interpolation",
             kind: str = "linear",
             **kwargs):
        """
        Construit une courbe ZC selon la méthode spécifiée.

        :param date: date de valorisation
        :param method: 'interpolation', 'nelson-siegel', 'svensson', 'vasicek'
        :param kind: type d'interpolation si applicable
        :param **kwargs : ->
                        initial_guess: paramètres initiaux de calibration (modèles paramétriques)
                        pas de temps (modèles stochastiques)
                        n_steps: nombre d'étapes (modèles stochastiques)
        :return: instance d'une courbe implémentant AbstractYieldCurve
        """
        method = method.lower()
        maturities, rates = self.get_maturity_and_rates(date)
        builder = ZeroCouponCurveBuilder(maturities, rates)
        return builder.build_curve(method=method,**kwargs)


if __name__ == "__main__":

    from data.management.data_retriever import DataRetriever
    from datetime import datetime
    from utils import tenor_to_years
    import matplotlib.pyplot as plt

    DR = DataRetriever("AMAZON")

    date = datetime(year=2023, month=10, day=1)
    curve = DR.get_risk_free_curve(date) / 100
    spot = DR.get_risk_free_index(date) / 100

    maturity = np.array([tenor_to_years(t) for t in curve.index])

    ns_guess = [0.02, -0.01, 0.01, 2.0]
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    zc = ZeroCouponCurveBuilder(maturity, curve.values)

    methods = {
        "Interpolation (cubic)": zc.build_curve(method="interpolation", kind="refined cubic"),
        "Nelson-Siegel": zc.build_curve(method="nelson-siegel", initial_guess=ns_guess),
        "Svensson": zc.build_curve(method="svensson", initial_guess=sv_guess),
        # "Vasicek": make_zc_curve(maturity, curve.values, method="vasicek", dt=1, n_steps=10),
    }

    mat = np.arange(maturity[0], maturity[-1], 0.01)
    # Tracé des courbes
    plt.figure(figsize=(10, 6))
    for label, zc_func in methods.items():
        rates_curve = zc_func.yield_curve_array(mat)
        plt.plot(mat, rates_curve, label=label)

    plt.scatter(maturity, curve.values, color='black', label='Données observées', zorder=5)
    plt.title("Courbes Zéro Coupon selon différentes méthodes")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux zéro-coupon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    import matplotlib.pyplot as plt

    DR = DataRetriever("AMAZON")

    date = datetime(year=2023, month=10, day=1)
    curve = DR.get_risk_free_curve(date) / 100
    spot = DR.get_risk_free_index(date) / 100

    maturity = np.array([tenor_to_years(t) for t in curve.index])

    ns_guess = [0.02, -0.01, 0.01, 2.0]
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    zc = ZeroCouponCurveBuilder(maturity, curve.values)

    methods = {
        "Interpolation (cubic)": zc.build_curve(method="interpolation", kind="refined cubic"),
        "Nelson-Siegel": zc.build_curve(method="nelson-siegel", initial_guess=ns_guess),
        "Svensson": zc.build_curve(method="svensson", initial_guess=sv_guess),
        # "Vasicek": make_zc_curve(maturity, curve.values, method="vasicek", dt=1, n_steps=10),
    }

    mat = np.arange(maturity[0], maturity[-1], 0.01)
    # Tracé des courbes
    plt.figure(figsize=(10, 6))
    for label, zc_func in methods.items():
        rates_curve = zc_func.yield_curve_array(mat)
        plt.plot(mat, rates_curve, label=label)

    plt.scatter(maturity, curve.values, color='black', label='Données observées', zorder=5)
    plt.title("Courbes Zéro Coupon selon différentes méthodes")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux zéro-coupon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()