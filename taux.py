import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, CubicSpline


class Taux:
    def __init__(self, maturities, rates):
        """
        Initialise la courbe de taux avec les maturités et les taux correspondants.
        :param maturities: liste ou tableau des maturités (ex. [1, 2, 3, 5, 7, 10])
        :param rates: liste ou tableau des taux associés à chaque maturité (ex. [0.01, 0.015, 0.017, 0.02, 0.022, 0.025])
        """
        self.maturities = np.array(maturities)
        self.rates = np.array(rates)

        # Création des fonctions d'interpolation pour méthode linéaire et cubique
        self.linear_interp = interp1d(self.maturities, self.rates, kind='linear', fill_value="extrapolate")
        self.cubic_interp = CubicSpline(self.maturities, self.rates, extrapolate=True)

    def get_rate(self, maturity, method="linear"):
        """
        Retourne le taux interpolé pour une maturité donnée.
        :param maturity: maturité pour laquelle on souhaite calculer le taux
        :param method: 'linear' pour une interpolation linéaire ou 'cubic' pour une interpolation cubique
        :return: taux interpolé pour la maturité demandée
        """
        if method == "linear":
            return self.linear_interp(maturity)
        elif method == "cubic":
            return self.cubic_interp(maturity)
        else:
            raise ValueError("Méthode d'interpolation non reconnue, choisissez 'linear' ou 'cubic'.")

class VasicekModel:
    def __init__(self, r0, kappa, theta, sigma):
        """
        Initialisation du modèle de Vasicek.
        :param r0: taux initial
        :param kappa: vitesse de réversion
        :param theta: niveau long terme
        :param sigma: volatilité
        """
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def bond_price(self, T):
        """
        Calcule le prix d'une obligation zéro coupon de maturité T.
        P(0, T) = A(T) * exp(-B(T) * r0)
        """
        B = (1 - np.exp(-self.kappa * T)) / self.kappa
        A = np.exp((self.theta - self.sigma ** 2 / (2 * self.kappa ** 2)) * (B - T) - (self.sigma ** 2 * B ** 2) / (
                    4 * self.kappa))
        return A * np.exp(-B * self.r0)

    def calibrate(self, maturities, market_prices, initial_params):
        """
        Calibre les paramètres (kappa, theta, sigma) du modèle afin de minimiser
        la somme des carrés des écarts entre les prix théoriques et les prix observés.
        :param maturities: liste des maturités pour lesquelles on a des prix de marché.
        :param market_prices: liste des prix d'obligations observés sur le marché.
        :param initial_params: estimation initiale pour [kappa, theta, sigma].
        """

        def calibration_error(params):
            kappa, theta, sigma = params
            # Crée un modèle temporaire avec les paramètres testés
            model = VasicekModel(self.r0, kappa, theta, sigma)
            # Erreur : somme des carrés des écarts
            errors = [model.bond_price(T) - mp for T, mp in zip(maturities, market_prices)]
            return np.sum(np.square(errors))

        result = minimize(calibration_error, initial_params, method='Nelder-Mead')
        self.kappa, self.theta, self.sigma = result.x
        return result.x




if __name__ == "__main__":
    # Données de marché fictives pour la courbe des taux
    taus = np.array([0.5, 1, 2, 3, 5, 7, 10])
    observed_yields = np.array([0.025, 0.027, 0.03, 0.032, 0.035, 0.037, 0.04])

    # Création du modèle avec des paramètres initiaux approximatifs
    ns_model = NelsonSiegelModel(beta0=0.03, beta1=-0.02, beta2=0.01, tau1=1.0)
    calibrated_params = ns_model.calibrate(taus, observed_yields, initial_guess=[0.03, -0.02, 0.01, 1.0])
    print(f"Paramètres calibrés (Nelson-Siegel) : beta0={calibrated_params[0]:.4f}, beta1={calibrated_params[1]:.4f}, beta2={calibrated_params[2]:.4f}, tau1={calibrated_params[3]:.4f}")
    # Affichage de quelques rendements théoriques calibrés
    yields = ns_model.yield_curve_array(taus)
    print("Taux théoriques calibrés pour les échéances :", yields)