import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List
from volatility.abstract_volatility import VolatilityModel
from volatility.calibration.svi_params import SVIParams, MarketDataPoint, SVICalibrationParams
from data.management.manage_bloomberg_data import OptionDataParser
import pandas as pd
import os
from scipy.interpolate import interp1d


class SVI(VolatilityModel):
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, rho: float = 0.1, m: float = 0.1, sigma: float = 0.1) -> None:
        """
        Initialise le modèle SVI avec les paramètres par défaut ou fournis.
        :param alpha: Paramètre alpha : comme niveau de la volatilité
        :param beta: Paramètre beta (strict pos.) : comme niveau & amplitude de courbe
        :param rho: Paramètre rho (inf à 1 en absolu) ; comme pentification de la partie gauche de la courbe (strikes faibles)
        :param m: Paramètre m ; décalage et ajustement de la pente à droite (strikes élevés)
        :param sigma: Paramètre sigma ; niveau de la volatilité et courbure globale
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def total_variance(self, log_moneyness: float) -> float:
        """
        Calcule la variance totale selon la formule SVI pour une moneyness donnée.
        :param log_moneyness: log(Strike/S)
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
        if total_var < 0 :
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
        et la volatilité implicite observée sur le marché, avec une pénalité pour les paramètres
        hors domaine. Utilise l'algorithme Nelder-Mead.
        :param calibration_params: Objet SVICalibrationParams contenant les données de marché et le spot.
        """
        opt_data = calibration_params.opt_data
        spot = calibration_params.spot

        # Définition des bornes pour chaque paramètre (à ajuster)
        amin, amax = 1e-5, 5.0        # Bornes pour alpha
        bmin, bmax = 1e-3, 1.0         # Bornes pour beta
        rho_min, rho_max = -0.999, 0.999   # Bornes pour rho
        m_min, m_max = -1.0, 1.0       # Bornes pour m (décalage)
        sigma_min, sigma_max = 0.01, 1.0  # Bornes pour sigma

        penalty_coef = 1e6  # Coefficient de pénalité

        def objective_function(x: np.ndarray) -> float:
            self.set_parameters(x.tolist())
            error_sum = 0.0
            # Calcul de l'erreur sur les données de marché
            for data in opt_data:
                svi_params = SVIParams(strike=data.strike, maturity=data.maturity, spot=spot)
                model_vol = self.get_volatility(svi_params) * 100  # conversion en %
                error = model_vol - data.implied_volatility
                error_sum += error**2

            # Ajout d'une pénalité pour chaque paramètre hors de sa plage autorisée
            # car on avait des NaN dans la surface de volatilité
            penalty = 0.0
            # Pour alpha
            if x[0] < amin:
                penalty += penalty_coef * (amin - x[0])**2
            elif x[0] > amax:
                penalty += penalty_coef * (x[0] - amax)**2
            # Pour beta
            if x[1] < bmin:
                penalty += penalty_coef * (bmin - x[1])**2
            elif x[1] > bmax:
                penalty += penalty_coef * (x[1] - bmax)**2
            # Pour rho
            if x[2] < rho_min:
                penalty += penalty_coef * (rho_min - x[2])**2
            elif x[2] > rho_max:
                penalty += penalty_coef * (x[2] - rho_max)**2
            # Pour m
            if x[3] < m_min:
                penalty += penalty_coef * (m_min - x[3])**2
            elif x[3] > m_max:
                penalty += penalty_coef * (x[3] - m_max)**2
            # Pour sigma
            if x[4] < sigma_min:
                penalty += penalty_coef * (sigma_min - x[4])**2
            elif x[4] > sigma_max:
                penalty += penalty_coef * (x[4] - sigma_max)**2

            return error_sum + penalty

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
                vol_surface[i, j] = self.get_volatility(params)

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

def convert_maturity(maturity_str):
    """
    Convertit une chaîne représentant la maturité en fraction d'année.
    Exemples:
      "1W" -> 1/52
      "5W" -> 5/52
      "1M" -> 1/12
      "10M" -> 10/12
      "1Y" -> 1.0
      "2Y" -> 2.0
    """
    maturity_str = maturity_str.strip().upper()
    if maturity_str.endswith('W'):
        try:
            weeks = float(maturity_str[:-1])
            return weeks / 52.0
        except:
            return np.nan
    elif maturity_str.endswith('M'):
        try:
            months = float(maturity_str[:-1])
            return months / 12.0
        except:
            return np.nan
    elif maturity_str.endswith('Y'):
        try:
            years = float(maturity_str[:-1])
            return years
        except:
            return np.nan
    else:
        try:
            return float(maturity_str)
        except:
            return np.nan


if __name__ == "__main__":
    # =============================================================
    # Option 1 : Utiliser les données Bloomberg (déjà au format long)
    # Assure-toi que le fichier Excel contient déjà les colonnes "maturity", "strike" et "iv".
    # Décommente cette section si tu travailles avec ce format.
    # =============================================================
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data_options"))
    # file_path = f"{DATA_PATH}/options_data_TSLA 2.xlsx"
    # df_options = OptionDataParser.prepare_option_data(file_path)
    # # On suppose que df_options contient déjà les colonnes "maturity", "strike" et "iv".
    # # Si nécessaire, convertir les types :
    # df_options["maturity"] = df_options["maturity"].astype(float)
    # df_options = df_options[df_options["maturity"] <= 1.0] # on garde seulement les maturités inférieur ou égale à 1 ans
    # df_options["strike"] = df_options["strike"].astype(float)
    # df_options["vol"] = df_options["vol"].astype(float)

    # =============================================================
    # Option 2 : Utiliser le fichier "clean_data_SPX.xlsx" (format wide)
    # La première colonne "Maturity" contient des maturités sous forme de chaînes ("1W", "1M", "2Y", etc.)
    # Les autres colonnes sont les strikes avec les valeurs de volatilité.
    # Décommente cette section si tu travailles avec ce format.
    # =============================================================
    file_path = f"{DATA_PATH}/clean_data_SPX.xlsx"
    df_options = pd.read_excel(file_path)
    df_options = df_options.melt(
        id_vars="Maturity",  # La colonne à conserver (maturité sous forme de chaîne)
        var_name="strike",  # Les anciennes colonnes deviennent la colonne "strike"
        value_name="iv"  # Les valeurs deviennent la colonne "iv"
    )
    # Conversion de "Maturity" en années à l'aide de la fonction convert_maturity
    df_options["Maturity"] = df_options["Maturity"].apply(convert_maturity)
    # on garde seulement les maturités inférieur ou égale à 1 ans
    df_options = df_options[df_options["Maturity"] <= 1.0]
    df_options["strike"] = df_options["strike"].astype(float)
    df_options["iv"] = df_options["iv"].astype(float)
    #
    # # Uniformisation des noms de colonnes pour la suite
    df_options.columns = ["maturity", "strike", "iv"]

    # =============================================================
    # Paramètres généraux pour la calibration
    # =============================================================
    # Choix d'un spot adapté pour le SPX
    spot = df_options['strike'].mean()  # ou une autre valeur représentative

    # Récupérer la liste des maturités uniques (en années)
    maturities = np.sort(df_options["maturity"].unique())

    # Dictionnaires pour stocker les modèles calibrés et les données de chaque slice
    svi_models = {}
    data_slices = {}

    # =============================================================
    # Calibration slice par slice
    # =============================================================
    for T in maturities:
        # Extraction des données pour la maturité T
        slice_df = df_options[df_options["maturity"] == T]
        data_slices[T] = slice_df.copy()

        # Construction de la liste des MarketDataPoint pour cette tranche
        opt_data_slice = [
            MarketDataPoint(
                strike=row['strike'],
                maturity=row['maturity'],
                implied_volatility=row['iv']
            )
            for _, row in slice_df.iterrows()
        ]

        # Création de la structure de calibration pour la slice
        calibration_params = SVICalibrationParams(opt_data=opt_data_slice, spot=spot)

        # Instanciation et calibration du modèle SVI pour la slice courante
        svi_model = SVI()
        try:
            svi_model.calibrate(calibration_params)
        except Exception as e:
            print(f"Calibration échouée pour la maturité {T}: {e}")
            continue

        svi_models[T] = svi_model


        # Affichage des paramètres calibrés pour vérification
        print(f"Maturité {T}:")
        print("  Alpha =", svi_model.alpha)
        print("  Beta  =", svi_model.beta)
        print("  Rho   =", svi_model.rho)
        print("  m     =", svi_model.m)
        print("  Sigma =", svi_model.sigma)
        print("-" * 40)

    # =============================================================
    # Visualisation : Tracé de la smile calibrée et des points de marché pour chaque slice
    # =============================================================
    strike_min = df_options["strike"].min()
    strike_max = df_options["strike"].max()
    strike_range = np.linspace(strike_min, strike_max, 100)


    for T, model in svi_models.items():
        # Calcul de la smile calibrée
        # On suppose que get_volatility renvoie la volatilité en décimal, et on multiplie par 100 pour l'exprimer en %
        smile_vols = [
            model.get_volatility(SVIParams(strike=k, maturity=T, spot=spot)) * 100
            for k in strike_range
        ]


        plt.figure(figsize=(8, 5))
        plt.plot(strike_range, smile_vols, 'b-', label=f"Smile calibrée (T = {T:.2f} an)")

        # Superposition des points de données de marché
        slice_df = data_slices[T]
        plt.scatter(slice_df["strike"], slice_df["iv"], color="red", marker="o", s=50, label="Données marché")

        plt.xlabel("Strike")
        plt.ylabel("Volatilité implicite (%)")
        plt.title(f"SVI Smile pour maturité T = {T:.2f} an")
        plt.legend()
        plt.grid(True)
        plt.show()

    # =============================================================
    # Construction d'une surface de volatilité globale
    # =============================================================
    # Pour chaque slice, extraire les paramètres calibrés
    maturities_calib = np.array(sorted(svi_models.keys()))
    alpha_arr = np.array([svi_models[T].alpha for T in maturities_calib])
    beta_arr = np.array([svi_models[T].beta for T in maturities_calib])
    rho_arr = np.array([svi_models[T].rho for T in maturities_calib])
    m_arr = np.array([svi_models[T].m for T in maturities_calib])
    sigma_arr = np.array([svi_models[T].sigma for T in maturities_calib])

    # Interpolation des paramètres par maturité (on utilise ici un lissage linéaire)
    alpha_interp = interp1d(maturities_calib, alpha_arr, kind="linear", fill_value="extrapolate")
    beta_interp = interp1d(maturities_calib, beta_arr, kind="linear", fill_value="extrapolate")
    rho_interp = interp1d(maturities_calib, rho_arr, kind="linear", fill_value="extrapolate")
    m_interp = interp1d(maturities_calib, m_arr, kind="linear", fill_value="extrapolate")
    sigma_interp = interp1d(maturities_calib, sigma_arr, kind="linear", fill_value="extrapolate")


    # Définition d'une fonction globale pour calculer la volatilité via l'interpolation
    def global_volatility(T, strike, spot):
        # Récupère les paramètres interpolés pour la maturité T
        a = float(alpha_interp(T))
        b = float(beta_interp(T))
        rho = float(rho_interp(T))
        m_val = float(m_interp(T))
        sigma_val = float(sigma_interp(T))
        log_moneyness = np.log(strike / spot)
        total_var = a + b * (rho * (log_moneyness - m_val) + np.sqrt((log_moneyness - m_val) ** 2 + sigma_val ** 2))
        # if total_var <= 0 or T <= 0:
        #     return 0.0 # ou 0.0, ou lever une exception
        # La volatilité annualisée est donnée par sqrt(total variance / T)
        vol = np.sqrt(total_var / T)
        return vol


    # Construction d'une grille en maturité et strike pour tracer la surface
    strike_min = df_options["strike"].min()
    strike_max = df_options["strike"].max()
    strikes_grid = np.linspace(strike_min, strike_max, 50)
    maturities_grid = np.linspace(maturities_calib.min(), maturities_calib.max(), 50)
    strikes_mesh, maturities_mesh = np.meshgrid(strikes_grid, maturities_grid)

    # Calcul de la surface de volatilité
    vol_surface = np.zeros_like(strikes_mesh)
    for i in range(maturities_mesh.shape[0]):
        for j in range(maturities_mesh.shape[1]):
            T_val = maturities_mesh[i, j]
            strike_val = strikes_mesh[i, j]
            # global_volatility renvoie la vol en décimal, multiplie par 100 pour obtenir le %
            vol_surface[i, j] = global_volatility(T_val, strike_val, spot) * 100

    print("Nombre de NaN dans vol_surface :", np.isnan(vol_surface).sum())
    print("Nombre de Inf dans vol_surface :", np.isinf(vol_surface).sum())
    # Tracé de la surface 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(strikes_mesh, maturities_mesh, vol_surface, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturité (années)')
    ax.set_zlabel('Volatilité implicite (%)')
    ax.set_title("Surface de volatilité SVI globale interpolée")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()