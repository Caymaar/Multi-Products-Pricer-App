import numpy as np
from scipy.optimize import minimize
import pandas as pd
import plotly.graph_objects as go
from product import ZeroCouponBond

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
    np.random.seed(123)
    data = pd.read_excel("../data_taux/RateCurve_temp.xlsx")
    maturities = data['Matu'].values
    observed_yields = data['Rate'].values
    # Calcul des prix de marché pour des obligations zéro coupon
    # On utilise la formule P_mkt = exp(-r * T)
    market_prices = np.exp(-observed_yields * maturities)

    # Initialisation du modèle avec r0 égal au premier taux observé
    r0 = observed_yields[0]
    # Estimation initiale pour [kappa, theta, sigma]
    initial_params = [0.1, 0.05, 0.01]

    model = VasicekModel(r0, *initial_params)
    calibrated_params = model.calibrate(maturities, market_prices, initial_params)

    print("Paramètres calibrés (Vasicek) :")
    print(f"kappa = {calibrated_params[0]:.4f}, theta = {calibrated_params[1]:.4f}, sigma = {calibrated_params[2]:.4f}")

    # Calcul des prix théoriques à partir du modèle calibré
    theoretical_prices = np.array([model.bond_price(T) for T in maturities])

    # Calcul des résidus (écarts entre les prix théoriques et les prix de marché)
    residuals = theoretical_prices - market_prices

    # Calcul de l'erreur quadratique moyenne (MSE)
    mse = np.mean(np.square(residuals))
    print("Erreur quadratique moyenne (MSE) :", mse)

    # --- Visualisation des prix de marché vs théoriques avec Plotly ---

    fig_prices = go.Figure()

    # Ajout des points de prix de marché
    fig_prices.add_trace(go.Scatter(
        x=maturities,
        y=market_prices,
        mode='markers',
        name="Prix de marché",
        marker=dict(color='blue')
    ))

    # Ajout de la courbe théorique (modèle Vasicek)
    fig_prices.add_trace(go.Scatter(
        x=maturities,
        y=theoretical_prices,
        mode='lines',
        name="Prix théoriques (Vasicek)",
        line=dict(color='red')
    ))

    fig_prices.update_layout(
        title="Comparaison des prix d'obligations : Marché vs Théorie",
        xaxis_title="Maturité (années)",
        yaxis_title="Prix de l'obligation",
        template="plotly_white"
    )

    fig_prices.show()

    # --- Visualisation des résidus avec Plotly ---

    fig_residuals = go.Figure()

    # Ajout des résidus
    fig_residuals.add_trace(go.Scatter(
        x=maturities,
        y=residuals,
        mode='markers+lines',
        name="Résidus",
        marker=dict(color='black')
    ))

    # Ajout d'une ligne horizontale à y=0 pour référence
    fig_residuals.add_shape(
        type="line",
        x0=min(maturities),
        y0=0,
        x1=max(maturities),
        y1=0,
        line=dict(color="gray", dash="dash")
    )

    fig_residuals.update_layout(
        title="Résidus de la calibration du modèle de Vasicek",
        xaxis_title="Maturité (années)",
        yaxis_title="Écart (Théorique - Marché)",
        template="plotly_white"
    )

    fig_residuals.show()