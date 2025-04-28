# from rate.interpolation import RateInterpolation
# from rate.nelson_siegel import NelsonSiegelModel
# from rate.svensson import SvenssonModel
# from rate.vasicek import VasicekModel


# def bootstrap_zero_curve(taus: np.ndarray, swap_rates: np.ndarray, freq: int = 1):
#     """
#     Bootstrapping d'une courbe zéro-coupon à partir de swap par rates.

#     :param tenors: liste de maturités ['1Y','2Y',...]
#     :param swap_rates: array de par rates (en décimal, p.ex. 0.02 pour 2%)
#     :param freq: fréquence des paiements fixes par an (1=annuel, 2=semi-annuel...)
#     :return: tuple (discount_factors, zero_rates)
#     """

#     # Calcule des year fractions entre paiements
#     # dt_i = tau_i - tau_{i-1}, avec tau_0 = 0
#     year_fracs = np.diff(np.concatenate(([0.0], taus)))

#     n = len(taus)
#     dfs = np.zeros(n)

#     # Bootstrapping itératif
#     for i in range(n):
#         c = swap_rates[i]
#         dt = year_fracs[i]

#         if i == 0:
#             # Premier point : simple discount
#             dfs[i] = 1.0 / (1.0 + c * dt)
#         else:
#             # Somme des flux fixes actualisés jusqu'à i-1
#             fixed_pv = sum(c * year_fracs[j] * dfs[j] for j in range(i))
#             # Equation par rate : c * sum_{j=1}^i DF_j * dt_j + DF_i = 1
#             dfs[i] = (1.0 - fixed_pv) / (1.0 + c * dt)

#     # Taux zéro-coupon
#     zero_rates = -np.log(dfs) / taus

#     return zero_rates

# def make_zc_curve(method, *args, kind='linear', **kwargs):
#     """
#     Fabrique une fonction de taux zéro-coupon en fonction de la méthode choisie.

#     :param method: 'interpolation', 'nelson-siegel', 'svensson', 'vasicek'
#     :param args: arguments positionnels spécifiques à chaque méthode
#     :param kind: 'linear', 'cubic' ou 'refined cubic' (si interpolation)
#     :param kwargs: arguments nommés spécifiques à la méthode (ex: a, b, sigma, r0 pour vasicek)
#     :return: fonction zc_curve(t)
#     """

#     method = method.lower()

#     if method == 'interpolation':
#         # args = (maturities, observed_yields)
#         if len(args) != 2:
#             raise ValueError("Interpolation requiert (maturities, observed_yields)")
#         maturities, observed_yields = args
#         interp = RateInterpolation(maturities, observed_yields, kind=kind)
#         return lambda t: interp.yield_value(t)

#     elif method == 'nelson-siegel':
#         # args = (maturities, observed_yields, initial_guess)
#         if len(args) != 3:
#             raise ValueError("Nelson-Siegel requiert (maturities, observed_yields, initial_guess)")
#         maturities, observed_yields, initial_guess = args
#         model = NelsonSiegelModel(*initial_guess)
#         model.calibrate(maturities, observed_yields, initial_guess)
#         return lambda t: model.yield_value(t)

#     elif method == 'svensson':
#         # args = (maturities, observed_yields, initial_guess)
#         if len(args) != 3:
#             raise ValueError("Svensson requiert (maturities, observed_yields, initial_guess)")
#         maturities, observed_yields, initial_guess = args
#         model = SvenssonModel(*initial_guess, maturities, observed_yields, initial_guess)
#         return lambda t: model.yield_value(t)

#     elif method == 'vasicek':
#         # kwargs attendus : soit from_data, soit a, b, sigma, r0 (optionnel)
#         if "from_data" in kwargs:
#             model = VasicekModel.calibrate_from_file(kwargs["from_data"])
#         else:
#             required_keys = ['observed_yields', 'dt', 'n_steps']
#             for key in required_keys:
#                 if key not in kwargs:
#                     raise ValueError(f"Paramètre requis manquant pour Vasicek : {key}")
#             model = VasicekModel.calibrate_dt(
#                 observed_yields=kwargs["observed_yields"],
#                 dt=kwargs["dt"],
#                 n_steps=kwargs["n_steps"]
#             )
#         return lambda t: model.yield_value(t)

#     else:
#         raise ValueError(f"Méthode inconnue : {method}")

# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Courbe fictive pour les tests
#     maturities = np.array([1, 2, 3, 5, 7, 10])
#     rates = np.array([0.015, 0.017, 0.018, 0.020, 0.022, 0.025])
#     t_values = np.linspace(0.1, 10, 100)

#     # Définition du pas de temps et du nombre de pas
#     dt = 1
#     n_steps = 10

#     # Initial guesses (exemples arbitraires)
#     ns_guess = [0.02, -0.01, 0.01, 2.0]
#     sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    # methods = {
    #     "Interpolation (cubic)": make_zc_curve("interpolation", maturities, rates, kind="refined cubic"),
    #     "Nelson-Siegel": make_zc_curve("nelson-siegel", maturities, rates, ns_guess),
    #     "Svensson": make_zc_curve("svensson", maturities, rates, sv_guess),
    #             "Vasicek": make_zc_curve(
    #         "vasicek",
    #         observed_yields=rates,
    #         dt=dt,
    #         n_steps=n_steps
    #     )
    #     # "Vasicek (from file)": make_zc_curve("vasicek", from_data="chemin/vers/fichier.csv")  # à tester si fichier réel
    # }

    # # Tracé des courbes
    # plt.figure(figsize=(10, 6))
    # for label, zc_func in methods.items():
    #     rates_curve = [zc_func(t) for t in t_values]
    #     plt.plot(t_values, rates_curve, label=label)

    # plt.scatter(maturities, rates, color='black', label='Données observées', zorder=5)
    # plt.title("Courbes Zéro Coupon selon différentes méthodes")
    # plt.xlabel("Maturité (années)")
    # plt.ylabel("Taux zéro-coupon")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

import numpy as np
from rate.interpolation import RateInterpolation
from rate.nelson_siegel import NelsonSiegelModel
from rate.svensson import SvenssonModel
from rate.vasicek import VasicekModel


class ZeroCouponCurveBuilder:
    """
    Constructeur de courbe zéro-coupon à partir de taux par swap.
    Bootstrapping puis construction de la courbe selon la méthode choisie.
    """
    def __init__(self, taus: np.ndarray, swap_rates: np.ndarray, freq: int = 1):
        self.taus = np.array(taus)
        self.swap_rates = np.array(swap_rates)
        self.freq = freq
        self.dfs = None
        self.zero_rates = None
        self.dt = np.diff(np.concatenate(([0.0], self.taus)))[0]
        self._bootstrap()

    def _bootstrap(self):
        """
        Calcul des discount factors et des taux zéro-coupon par bootstrapping.
        """
        year_fracs = np.diff(np.concatenate(([0.0], self.taus)))
        n = len(self.taus)
        dfs = np.zeros(n)
        for i in range(n):
            c = self.swap_rates[i]
            dt = year_fracs[i]
            if i == 0:
                dfs[i] = 1.0 / (1.0 + c * dt)
            else:
                fixed_pv = np.sum(self.swap_rates[:i] * year_fracs[:i] * dfs[:i])
                dfs[i] = (1.0 - fixed_pv) / (1.0 + c * dt)
        self.dfs = dfs
        self.zero_rates = -np.log(dfs) / self.taus

    def build_curve(self,
                    method: str = 'interpolation',
                    kind: str = 'linear',
                    initial_guess: list = None,
                    dt: float = None,
                    n_steps: int = None):
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
            interp = RateInterpolation(self.taus, self.zero_rates, kind=kind)
            return InterpolatedZCCurve(interp)

        elif method == 'nelson-siegel':
            if initial_guess is None:
                raise ValueError("initial_guess requis pour Nelson-Siegel")
            model = NelsonSiegelModel.calibrate(self.taus, self.zero_rates, initial_guess)
            return ParametricZCCurve(model)

        elif method == 'svensson':
            if initial_guess is None:
                raise ValueError("initial_guess requis pour Svensson")
            model = SvenssonModel.calibrate(self.taus, self.zero_rates, initial_guess)
            return ParametricZCCurve(model)

        elif method == 'vasicek':
            if dt is None or n_steps is None:
                raise ValueError("dt et n_steps requis pour Vasicek")
            model = VasicekModel.calibrate(self.zero_rates, self.dt, n_steps)
            return ParametricZCCurve(model)

        else:
            raise ValueError(f"Méthode inconnue : {method}")


class InterpolatedZCCurve:
    """
    Courbe zéro-coupon basée sur interpolation.
    """
    def __init__(self, interpolator: RateInterpolation):
        self.interp = interpolator

    def zero_rate(self, t: float) -> float:
        return self.interp.yield_value(t)

    def zero_curve(self, maturities: np.ndarray) -> np.ndarray:
        return self.interp.yield_curve_array(maturities)


class ParametricZCCurve:
    """
    Courbe zéro-coupon basée sur modèle paramétrique (Nelson-Siegel, Svensson, Vasicek).
    """
    def __init__(self, model):
        self.model = model

    def zero_rate(self, t: float) -> float:
        return self.model.yield_value(t)

    def zero_curve(self, maturities: np.ndarray) -> np.ndarray:
        return np.array([self.zero_rate(t) for t in maturities])


def make_zc_curve(taus: np.ndarray,
                  swap_rates: np.ndarray,
                  method: str = 'interpolation',
                  **kwargs):
    """
    Factory pour construire une courbe zéro-coupon.

    :param taus: maturités (en années)
    :param swap_rates: par rates correspondants
    :param method: méthode de construction
    :param kwargs: kind, initial_guess, dt, n_steps
    :return: objet courbe avec zero_rate(t) et zero_curve(maturities)
    """
    builder = ZeroCouponCurveBuilder(taus, swap_rates, freq=kwargs.get('freq', 1))
    return builder.build_curve(method=method,
                               kind=kwargs.get('kind', 'linear'),
                               initial_guess=kwargs.get('initial_guess'),
                               n_steps=kwargs.get('n_steps'))


if __name__ == "__main__":

    from data.management.data_retriever import DataRetriever
    from rate.interpolation import RateInterpolation
    from rate.bootstrap import bootstrap_zero_curve
    from utils import tenor_to_years
    import matplotlib.pyplot as plt

    DR = DataRetriever("AMAZON")

    date = "2023-10-01"
    curve = DR.get_risk_free_curve(date) / 100
    spot = DR.get_risk_free_index(date) /100

    maturity = np.array([tenor_to_years(t) for t in curve.index])

    ns_guess = [0.02, -0.01, 0.01, 2.0]
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]
    
    methods = {
        "Interpolation (cubic)": make_zc_curve(maturity, curve.values, method="interpolation", kind="refined cubic"),
        "Nelson-Siegel": make_zc_curve(maturity, curve.values, method="nelson-siegel", initial_guess=ns_guess),
        "Svensson": make_zc_curve(maturity, curve.values, method="svensson", initial_guess=sv_guess),
        #"Vasicek": make_zc_curve(maturity, curve.values, method="vasicek", dt=1, n_steps=10),
        }
    


    mat= np.arange(maturity[0], maturity[-1], 0.01)
    # Tracé des courbes
    plt.figure(figsize=(10, 6))
    for label, zc_func in methods.items():
        rates_curve = zc_func.zero_curve(mat)
        plt.plot(mat, rates_curve, label=label)

    plt.scatter(maturity, curve.values, color='black', label='Données observées', zorder=5)
    plt.title("Courbes Zéro Coupon selon différentes méthodes")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux zéro-coupon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
