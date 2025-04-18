import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from abstract_taux import AbstractRateModel

class NelsonSiegelModel(AbstractRateModel):
    def __init__(self, beta0, beta1, beta2, lambda1):
        """
        Initialisation du modèle Nelson-Siegel.
        :param beta0: niveau de long terme
        :param beta1: composante de la pente
        :param beta2: composante de la courbure
        :param lambda1: facteur de déclin
        """
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda1 = lambda1

    def yield_value(self, t):
        """
        Calcule le taux pour une échéance donnée t.
        """
        # Gestion de la division par zéro pour t == 0
        if t == 0:
            return self.beta0 + self.beta1 + 0.5 * self.beta2
        factor = t / self.lambda1
        term1 = (1 - np.exp(-factor)) / factor
        term2 = term1 - np.exp(-factor) - np.exp(-factor)
        return self.beta0 + self.beta1 * term1 + self.beta2 * term2

    def calibrate(self, maturities, observed_yields, initial_guess):
        """
        Calibre les paramètres du modèle Nelson-Siegel.

        :param maturities: Tableau des échéances (obligatoire pour ce modèle)
        :param observed_yields: Tableau des rendements observés
        :param initial_guess: Estimation initiale des paramètres [beta0, beta1, beta2, lambda1]
        :return: Les paramètres calibrés
        """
        if maturities is None or observed_yields is None or initial_guess is None:
            raise ValueError(
                "Pour Nelson-Siegel, 'maturities', 'observed_yields' et 'initial_guess' doivent être fournis.")

        # Définition de la fonction objectif (somme des carrés des erreurs)
        def objective(params):
            beta0, beta1, beta2, lambda1 = params
            y_pred = []
            for t in maturities:
                if t == 0:
                    y = beta0 + beta1 + 0.5 * beta2
                else:
                    factor = t / lambda1
                    term1 = (1 - np.exp(-factor)) / factor
                    term2 = term1 - np.exp(-factor) - np.exp(-factor)
                    y = beta0 + beta1 * term1 + beta2 * term2
                y_pred.append(y)
            y_pred = np.array(y_pred)
            return np.sum((y_pred - observed_yields) ** 2)

        # Exécution de l'optimisation avec Nelder-Mead
        res = minimize(objective, initial_guess, method='Nelder-Mead')
        self.beta0, self.beta1, self.beta2, self.lambda1 = res.x
        return res.x

    def plot_fit(self, maturities, observed_yields, n_points=100, title="Calibration du modèle Nelson-Siegel"):
        """
        Trace la courbe ajustée par le modèle ainsi que les données observées.

        :param maturities: tableau des échéances observées
        :param observed_yields: tableau des rendements observés
        :param n_points: nombre de points pour tracer la courbe lissée
        :param title: titre du graphique
        """
        maturities_fine = np.linspace(min(maturities), max(maturities), n_points)
        fitted_yields = self.yield_curve_array(maturities_fine)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=maturities,
            y=observed_yields,
            mode='markers',
            name="Données observées",
            marker=dict(color='black')
        ))

        fig.add_trace(go.Scatter(
            x=maturities_fine,
            y=fitted_yields,
            mode='lines',
            name="Courbe Nelson-Siegel"
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Échéance (t)",
            yaxis_title="Rendement",
            template="plotly_white"
        )

        fig.show()

if __name__ == "__main__":
    np.random.seed(123)
    data = pd.read_excel("../data_taux/RateCurve_temp.xlsx")
    maturities_raw = data['Matu'].values
    observed_yields = data['Rate'].values

    # --- Calibration des modèles ---
    initial_guess_ns = [4.0, -1.0, 0.5, 2]
    ns_model = NelsonSiegelModel(beta0=initial_guess_ns[0],
                                 beta1=initial_guess_ns[1],
                                 beta2=initial_guess_ns[2],
                                 lambda1=initial_guess_ns[3])
    params_ns = ns_model.calibrate(maturities_raw, observed_yields, initial_guess_ns)

    print("\nParamètres calibrés (Nelson-Siegel) :")
    print(
        f"beta0 = {params_ns[0]:.4f}, beta1 = {params_ns[1]:.4f}, beta2 = {params_ns[2]:.4f}, lambda1 = {params_ns[3]:.4f}")

    ns_model.plot_fit(maturities_raw, observed_yields, title="Calibration du modèle Nelson-Siegel")