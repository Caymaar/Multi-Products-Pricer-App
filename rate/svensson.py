import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from abstract_taux import AbstractRateModel

class SvenssonModel(AbstractRateModel):
    def __init__(self, beta0, beta1, beta2, beta3, lambda1, lambda2, maturities, observed_yields, initial_guess):
        """
        Initialisation du modèle Svensson.

        Paramètres:
        - beta0 : niveau de long terme
        - beta1 : composante de la pente
        - beta2 : première composante de la courbure
        - beta3 : deuxième composante de la courbure
        - lambda1  : premier facteur de déclin
        - lambda2  : deuxième facteur de déclin
        """
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.params = self.calibrate(maturities,observed_yields,initial_guess)

    def yield_value(self, t):
        """
        Calcule le taux pour une échéance donnée t selon le modèle Svensson.

        Pour t == 0, on utilise la limite :
        y(0) = beta0 + beta1 + beta2/2 + beta3/2
        """
        if t == 0:
            return self.beta0 + self.beta1 + 0.5 * (self.beta2 + self.beta3)

        factor1 = t / self.lambda1
        factor2 = t / self.lambda2

        term1 = (1 - np.exp(-factor1)) / factor1
        term2 = term1 - np.exp(-factor1)
        term3 = (1 - np.exp(-factor2)) / factor2 #- np.exp(-factor2)

        return self.beta0 + self.beta1 * term1 + self.beta2 * term2 + self.beta3 * term3

    def calibrate(self, maturities, observed_yields, initial_guess):
        """
        Calibre les paramètres du modèle Svensson.

        :param maturities: Tableau des échéances (obligatoire pour ce modèle)
        :param observed_yields: Tableau des rendements observés
        :param initial_guess: Estimation initiale des paramètres [beta0, beta1, beta2, beta3, lambda1, lambda2]
        :return: Les paramètres calibrés
        """
        if maturities is None or observed_yields is None or initial_guess is None:
            raise ValueError("Pour Svensson, 'maturities', 'observed_yields' et 'initial_guess' doivent être fournis.")

        def objective(params):
            beta0, beta1, beta2, beta3, lambda1, lambda2 = params
            error = 0.0
            for t, obs in zip(maturities, observed_yields):
                if t == 0:
                    y_model = beta0 + beta1 + 0.5 * (beta2 + beta3)
                else:
                    factor1 = t / lambda1
                    factor2 = t / lambda2

                    term1 = (1 - np.exp(-factor1)) / factor1
                    term2 = term1 - np.exp(-factor1)
                    term3 = (1 - np.exp(-factor2)) / factor2 #- np.exp(-factor2)
                    y_model = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3
                error += (y_model - obs) ** 2
            return error

        res = minimize(objective, initial_guess, method='Nelder-Mead')
        self.beta0, self.beta1, self.beta2, self.beta3, self.lambda1, self.lambda2 = res.x
        return res.x

    def plot_fit(self, maturities, observed_yields, n_points=100, title="Calibration du modèle Svensson"):
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
            name="Courbe Svensson"
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
    # --- Chargement et nettoyage des données ---
    data = pd.read_excel("../data_taux/RateCurve_temp.xlsx")

    maturities_raw = data['Matu'].values
    observed_yields = data['Rate'].values

    # --- Calibration du modèle Svensson ---
    initial_guess_sv = [2.5, -1.0, 0.5, 0.3, 1.0, 2.0]
    svensson_model = SvenssonModel(beta0=initial_guess_sv[0], beta1=initial_guess_sv[1],
                                   beta2=initial_guess_sv[2], beta3=initial_guess_sv[3],
                                   lambda1=initial_guess_sv[4], lambda2=initial_guess_sv[5],
                                   maturities=maturities_raw, observed_yields=observed_yields,
                                   initial_guess=initial_guess_sv)
    params_sv = svensson_model.params
    print("\nParamètres calibrés (Svensson) :")
    print(f"beta0 = {params_sv[0]:.4f}, beta1 = {params_sv[1]:.4f}, beta2 = {params_sv[2]:.4f}, "
          f"beta3 = {params_sv[3]:.4f}, lambda1 = {params_sv[4]:.4f}, lambda2 = {params_sv[5]:.4f}")

    # --- Affichage des courbes de taux (plot fit) ---
    svensson_model.plot_fit(maturities_raw, observed_yields)
