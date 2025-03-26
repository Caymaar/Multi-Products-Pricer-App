import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from abstract_taux import AbstractYieldCurve


class NelsonSiegelModel(AbstractYieldCurve):
    def __init__(self, beta0, beta1, beta2, tau1):
        """
        Initialisation du modèle Nelson-Siegel.
        :param beta0: niveau de long terme
        :param beta1: composante de la pente
        :param beta2: composante de la courbure
        :param tau1: facteur de déclin
        """
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau1 = tau1

    def yield_curve(self, tau):
        """
        Calcule le taux pour une échéance donnée tau.
        """
        # Gestion de la division par zéro pour tau == 0
        if tau == 0:
            return self.beta0 + self.beta1 + 0.5 * self.beta2
        factor = tau / self.tau1
        term1 = (1 - np.exp(-factor)) / factor
        term2 = term1 - np.exp(-factor)
        return self.beta0 + self.beta1 * term1 + self.beta2 * term2

    def yield_curve_array(self, taus):
        """
        Calcule la courbe des taux pour un tableau d'échéances.
        """
        return np.array([self.yield_curve(tau) for tau in taus])

    def calibrate(self, taus, observed_yields, initial_guess):
        """
        Calibre les paramètres (beta0, beta1, beta2, tau1) du modèle en minimisant
        l'erreur quadratique entre les rendements observés et la courbe théorique,
        en utilisant l'algorithme Nelder-Mead.

        :param taus: tableau des échéances
        :param observed_yields: tableau des rendements observés
        :param initial_guess: estimation initiale pour [beta0, beta1, beta2, tau1]
        :return: les paramètres calibrés
        """

        # Définition de la fonction objectif (somme des carrés des erreurs)
        def objective(params):
            beta0, beta1, beta2, tau1 = params
            y_pred = []
            for tau in taus:
                if tau == 0:
                    y = beta0 + beta1 + 0.5 * beta2
                else:
                    factor = tau / tau1
                    term1 = (1 - np.exp(-factor)) / factor
                    term2 = term1 - np.exp(-factor)
                    y = beta0 + beta1 * term1 + beta2 * term2
                y_pred.append(y)
            y_pred = np.array(y_pred)
            return np.sum((y_pred - observed_yields) ** 2)

        # Exécution de l'optimisation avec Nelder-Mead
        res = minimize(objective, initial_guess, method='Nelder-Mead')
        self.beta0, self.beta1, self.beta2, self.tau1 = res.x
        return res.x

    def plot_fit(self, taus, observed_yields, n_points=100, title="Calibration du modèle Nelson-Siegel"):
        """
        Trace la courbe ajustée par le modèle ainsi que les données observées.

        :param taus: tableau des échéances observées
        :param observed_yields: tableau des rendements observés
        :param n_points: nombre de points pour tracer la courbe lissée
        :param title: titre du graphique
        """
        taus_fine = np.linspace(min(taus), max(taus), n_points)
        fitted_yields = self.yield_curve_array(taus_fine)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=taus,
            y=observed_yields,
            mode='markers',
            name="Données observées",
            marker=dict(color='black')
        ))

        fig.add_trace(go.Scatter(
            x=taus_fine,
            y=fitted_yields,
            mode='lines',
            name="Courbe Nelson-Siegel"
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Échéance (tau)",
            yaxis_title="Rendement",
            template="plotly_white"
        )

        fig.show()

if __name__ == "__main__":
    np.random.seed(123)
    data = pd.read_excel("../data_taux/RateCurve_temp.xlsx")
    taus_raw = data['Matu'].values
    observed_yields = data['Rate'].values

    # --- Calibration des modèles ---
    initial_guess_ns = [4.0, -1.0, 0.5, 2]
    ns_model = NelsonSiegelModel(beta0=initial_guess_ns[0],
                                 beta1=initial_guess_ns[1],
                                 beta2=initial_guess_ns[2],
                                 tau1=initial_guess_ns[3])
    params_ns = ns_model.calibrate(taus_raw, observed_yields, initial_guess_ns)

    print("\nParamètres calibrés (Nelson-Siegel) :")
    print(
        f"beta0 = {params_ns[0]:.4f}, beta1 = {params_ns[1]:.4f}, beta2 = {params_ns[2]:.4f}, tau1 = {params_ns[3]:.4f}")

    ns_model.plot_fit(taus_raw, observed_yields, title="Calibration du modèle Nelson-Siegel")