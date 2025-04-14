import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from stochastic_process.ou_process import OUProcess

class VasicekModel:
    def __init__(self, r0):
        """
        Initialisation du modèle de Vasicek sans paramètres initialisés.
        :param r0: taux initial
        """
        self.r0 = r0
        self.kappa = None  # Non initialisé
        self.theta = None  # Non initialisé
        self.sigma = None  # Non initialisé

    def calibrate(self, observed_rates, maturities):
        """
        Calibre les paramètres du modèle de Vasicek à partir des taux observés et des maturités.
        :param observed_rates: Liste des taux observés
        :param maturities: Liste des maturités
        :return: Aucun retour mais les paramètres sont stockés dans l'objet
        """
        # Calcul des différences de maturité pour obtenir les delta t
        dt_values = np.diff(maturities)

        # Variables pour la régression : X = r(t), Y = r(t+dt)
        X = observed_rates[:-1]  # r(t)
        Y = observed_rates[1:]   # r(t+dt)

        # Régression linéaire de Y sur X
        X_with_const = sm.add_constant(X)  # Ajouter une constante (intercept) pour la régression
        model = sm.OLS(Y, X_with_const)
        results = model.fit()

        # Coefficients estimés de la régression
        a_est, b_est = results.params

        # Estimation des paramètres du modèle de Vasicek

        # Estimation de kappa à partir de a
        self.kappa = (1 - a_est) / dt_values.mean()

        # Estimation de theta à partir de b et kappa
        self.theta = b_est / (self.kappa * dt_values.mean())

        # Estimation de sigma à partir de l'erreur de la régression
        residuals = results.resid
        self.sigma = np.std(residuals) / np.sqrt(dt_values.mean())

    def get_parameters(self):
        """
        Retourne les paramètres calibrés (kappa, theta, sigma).
        """
        return self.kappa, self.theta, self.sigma

    def get_rate_at_maturity(self, maturities, n_steps=1000):
        """
        Utilise le modèle de Vasicek pour obtenir un taux à une maturité spécifique
        via un processus d'Ornstein-Uhlenbeck simulé.

        :param maturities: Maturités disponibles
        :param n_steps: Nombre de pas de temps pour la simulation
        :return: Taux simulé à la maturité spécifiée
        """
        dt = (maturities[-1] - maturities[0]) / n_steps  # Ajustement du pas de temps

        ou_process = OUProcess(theta=self.theta, mu=self.r0, sigma=self.sigma,
                               initial_value=self.r0, n_paths=1, n_steps=n_steps, dt=dt)

        # Simuler un chemin et récupérer le taux à la maturité souhaitée
        path = ou_process.simulate_single_path()

        return np.linspace(maturities[0], maturities[-1], n_steps), path


if __name__ == "__main__":
    # Charger les données de la courbe de taux
    data = pd.read_excel("../data_taux/RateCurve_temp.xlsx")
    maturities = data['Matu'].values
    observed_rates = data['Rate'].values

    # Initialisation et calibration
    r0 = observed_rates[0]
    model = VasicekModel(r0)
    model.calibrate(observed_rates, maturities)

    # Récupération des paramètres calibrés
    kappa_est, theta_est, sigma_est = model.get_parameters()
    print(f"Paramètres calibrés : kappa = {kappa_est:.4f}, theta = {theta_est:.4f}, sigma = {sigma_est:.4f}")

    # Simulation des taux sur 1000 points pour une courbe fluide
    simulated_maturities, simulated_rates = model.get_rate_at_maturity(maturities, n_steps=1000)

    # Visualisation avec Plotly
    fig = go.Figure()

    # Taux du marché
    fig.add_trace(go.Scatter(
        x=maturities,
        y=observed_rates,
        mode='markers',
        name="Taux Observés",
        marker=dict(color='blue')
    ))

    # Courbe simulée Vasicek
    fig.add_trace(go.Scatter(
        x=simulated_maturities,
        y=simulated_rates,
        mode='lines',
        name="Courbe Vasicek (Simulée)",
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Courbe des taux : Observée vs Simulée (Vasicek)",
        xaxis_title="Maturité (années)",
        yaxis_title="Taux (%)",
        template="plotly_white"
    )

    fig.show()