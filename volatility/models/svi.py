# volatility/svi.py
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List
from volatility.abstract_volatility import VolatilityModel
from volatility.calibration.svi_params import SVIParams, MarketDataPoint, SVICalibrationParams
from data_management.manage_bloomberg_data import OptionDataParser


class SVI(VolatilityModel):
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, rho: float = 0.1, m: float = 0.1, sigma: float = 0.1) -> None:
        """
        Initialise le modèle SVI avec les paramètres par défaut ou fournis.
        :param alpha: Paramètre alpha
        :param beta: Paramètre beta
        :param rho: Paramètre rho
        :param m: Paramètre m (décalage)
        :param sigma: Paramètre sigma
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def total_variance(self, log_moneyness: float) -> float:
        """
        Calcule la variance totale selon la formule SVI pour une moneyness donnée.
        :param log_moneyness: log(S/Strike)
        :return: Variance totale.
        """
        return self.alpha + self.beta * (
            self.rho * (log_moneyness - self.m) +
            math.sqrt((log_moneyness - self.m)**2 + self.sigma**2)
        )

    def get_volatility(self, parameters: SVIParams) -> float:
        """
        Calcule la volatilité annualisée à partir du modèle SVI.
        :param parameters: Objet SVIParams contenant strike, maturity et spot.
        :return: Volatilité calculée.
        """
        log_moneyness = math.log(parameters.strike / parameters.spot)
        total_var = self.total_variance(log_moneyness)
        if total_var < 0:
            return 0.0

        return math.sqrt(total_var / parameters.maturity)

    def set_parameters(self, params: List[float]) -> None:
        """
        Met à jour les paramètres du modèle SVI.
        :param params: Liste [alpha, beta, rho, m, sigma]
        """
        self.alpha, self.beta, self.rho, self.m, self.sigma = params

    def calibrate(self, calibration_params: SVICalibrationParams) -> None:
        """
        Calibre le modèle SVI en minimisant l'erreur quadratique entre la volatilité modèle
        et la volatilité implicite observée sur le marché.
        :param calibration_params: Objet SVICalibrationParams contenant les données de marché et le spot.
        """
        opt_data = calibration_params.opt_data
        spot = calibration_params.spot

        def objective_function(x: np.ndarray) -> float:
            self.set_parameters(x.tolist())
            error_sum = 0.0
            for data in opt_data:
                svi_params = SVIParams(strike=data.strike, maturity=data.maturity, spot=spot)
                model_vol = self.get_volatility(svi_params) * 100  # en pourcentage
                error = model_vol - data.implied_volatility
                error_sum += error**2
            return error_sum

        initial_guess = [self.alpha, self.beta, self.rho, self.m, self.sigma]
        result = minimize(objective_function, x0=np.array(initial_guess), method='Nelder-Mead',
                          options={'xatol': 1e-1, 'maxiter': 100000})
        if result.success:
            self.set_parameters(result.x.tolist())
        else:
            raise Exception("SVI calibration failed: " + result.message)

    def plot_vol_surface(self, spot: float, strike_range: np.ndarray, maturity_range: np.ndarray) -> None:
        """
        Trace la surface de volatilité du modèle SVI sur une grille de strikes et maturités.

        :param spot: Prix actuel du sous-jacent.
        :param strike_range: Tableau numpy des strikes à tester.
        :param maturity_range: Tableau numpy des maturités à tester.
        """
        # Création d'une grille de strikes et maturités
        strikes, maturities = np.meshgrid(strike_range, maturity_range)
        vol_surface = np.zeros_like(strikes)

        # Calcul de la volatilité pour chaque combinaison (strike, maturité)
        for i in range(strikes.shape[0]):
            for j in range(strikes.shape[1]):
                params = SVIParams(strike=strikes[i, j], maturity=maturities[i, j], spot=spot)
                vol_surface[i, j] = self.get_volatility(params)  # en pourcentage

        # Création du plot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(strikes, maturities, vol_surface, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied volatility (%)')
        ax.set_title('SVI Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == "__main__":
    file_path = "C:/Users/admin/Desktop/cours dauphine/S2/Stru/data_options/options_data_TSLA 2.xlsx"
    df_options = OptionDataParser.prepare_option_data(file_path)

    # Convertir le DataFrame en liste de MarketDataPoint
    opt_data = [
        MarketDataPoint(strike=row['strike'], maturity=row['maturity'], implied_volatility=row['vol'])
        for _, row in df_options.iterrows()
    ]

    spot = 220.0
    calibration_params = SVICalibrationParams(opt_data=opt_data, spot=spot)
    svi_model = SVI()
    svi_model.calibrate(calibration_params)

    print("Paramètres calibrés:")
    print("Alpha =", svi_model.alpha)
    print("Beta  =", svi_model.beta)
    print("Rho   =", svi_model.rho)
    print("m     =", svi_model.m)
    print("Sigma =", svi_model.sigma)

    # Définir des plages pour strikes et maturités pour le plot
    strike_range = np.linspace(100, 300, 50)
    maturity_range = np.linspace(0.05, 2, 50)

    # Affichage de la surface de volatilité
    svi_model.plot_vol_surface(spot=spot, strike_range=strike_range, maturity_range=maturity_range)