from interpolation import TauxInterpolation
from nelson_siegel import NelsonSiegelModel
from svensson import SvenssonModel
from vasicek_v2 import VasicekModel

def make_zc_curve(method, *args, kind='linear', **kwargs):
    """
    Fabrique une fonction de taux zéro-coupon en fonction de la méthode choisie.

    :param method: 'interpolation', 'nelson-siegel', 'svensson', 'vasicek'
    :param args: arguments positionnels spécifiques à chaque méthode
    :param kind: 'linear', 'cubic' ou 'refined cubic' (si interpolation)
    :param kwargs: arguments nommés spécifiques à la méthode (ex: a, b, sigma, r0 pour vasicek)
    :return: fonction zc_curve(t)
    """

    method = method.lower()

    if method == 'interpolation':
        # args = (maturities, observed_yields)
        if len(args) != 2:
            raise ValueError("Interpolation requiert (maturities, observed_yields)")
        maturities, observed_yields = args
        interp = TauxInterpolation(maturities, observed_yields, kind=kind)
        return lambda t: interp.get_taux(t)

    elif method == 'nelson-siegel':
        # args = (maturities, observed_yields, initial_guess)
        if len(args) != 3:
            raise ValueError("Nelson-Siegel requiert (maturities, observed_yields, initial_guess)")
        maturities, observed_yields, initial_guess = args
        model = NelsonSiegelModel(*initial_guess)
        model.calibrate(maturities, observed_yields, initial_guess)
        return lambda t: model.yield_curve(t)

    elif method == 'svensson':
        # args = (maturities, observed_yields, initial_guess)
        if len(args) != 3:
            raise ValueError("Svensson requiert (maturities, observed_yields, initial_guess)")
        maturities, observed_yields, initial_guess = args
        model = SvenssonModel(*initial_guess, maturities, observed_yields, initial_guess)
        return lambda t: model.yield_curve(t)

    elif method == 'vasicek':
        # kwargs attendus : soit from_data, soit a, b, sigma, r0 (optionnel)
        if "from_data" in kwargs:
            model = VasicekModel.calibrate_from_file(kwargs["from_data"])
        else:
            required_keys = ['a', 'b', 'sigma']
            for key in required_keys:
                if key not in kwargs:
                    raise ValueError(f"Paramètre requis manquant pour Vasicek : {key}")
            model = VasicekModel(
                a=kwargs["a"],
                b=kwargs["b"],
                sigma=kwargs["sigma"],
                r0=kwargs.get("r0", 0.01),
            )
        return lambda t: model.zero_coupon_rate(t)

    else:
        raise ValueError(f"Méthode inconnue : {method}")

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Courbe fictive pour les tests
    maturities = np.array([1, 2, 3, 5, 7, 10])
    rates = np.array([0.015, 0.017, 0.018, 0.020, 0.022, 0.025])
    t_values = np.linspace(0.5, 10, 100)

    # Initial guesses (exemples arbitraires)
    ns_guess = [0.02, -0.01, 0.01, 2.0]
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    methods = {
        "Interpolation (cubic)": make_zc_curve("interpolation", maturities, rates, kind="refined cubic"),
        "Nelson-Siegel": make_zc_curve("nelson-siegel", maturities, rates, ns_guess),
        "Svensson": make_zc_curve("svensson", maturities, rates, sv_guess),
        "Vasicek": make_zc_curve("vasicek", a=0.1, b=0.03, sigma=0.01, r0=0.02),
        # "Vasicek (from file)": make_zc_curve("vasicek", from_data="chemin/vers/fichier.csv")  # à tester si fichier réel
    }

    # Tracé des courbes
    plt.figure(figsize=(10, 6))
    for label, zc_func in methods.items():
        rates_curve = [zc_func(t) for t in t_values]
        plt.plot(t_values, rates_curve, label=label)

    plt.scatter(maturities, rates, color='black', label='Données observées', zorder=5)
    plt.title("Courbes Zéro Coupon selon différentes méthodes")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux zéro-coupon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
