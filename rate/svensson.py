import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from rate.abstract_taux import AbstractRateModel

class SvenssonModel(AbstractRateModel):
    def __init__(self,
                 beta0: float,
                 beta1: float,
                 beta2: float,
                 beta3: float,
                 lambda1: float,
                 lambda2: float):
        """
        Initialisation du modèle Svensson.

        :param beta0: niveau de long terme
        :param beta1: composante de la pente
        :param beta2: première composante de la courbure
        :param beta3: deuxième composante de la courbure
        :param lambda1: premier facteur de déclin
        :param lambda2: deuxième facteur de déclin
        """
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def yield_value(self, t: float) -> float:
        """
        Calcule le taux pour une échéance donnée t selon le modèle Svensson.
        Pour t == 0, on utilise la limite analytique.
        """
        if t == 0:
            return self.beta0 + self.beta1 + 0.5 * (self.beta2 + self.beta3)

        f1 = t / self.lambda1
        f2 = t / self.lambda2
        term1 = (1 - np.exp(-f1)) / f1
        term2 = term1 - np.exp(-f1)
        term3 = (1 - np.exp(-f2)) / f2

        return self.beta0 + self.beta1 * term1 + self.beta2 * term2 + self.beta3 * term3

    @classmethod
    def calibrate(cls,
                  maturities: np.ndarray,
                  observed_yields: np.ndarray,
                  initial_guess: list) -> 'SvenssonModel':
        """
        Calibre les paramètres du modèle Svensson et renvoie une instance calibrée.

        :param maturities: échéances observées (en années)
        :param observed_yields: rendements observés
        :param initial_guess: estimation initiale [beta0, beta1, beta2, beta3, lambda1, lambda2]
        :return: instance SvenssonModel calibrée
        """
        if maturities is None or observed_yields is None or initial_guess is None:
            raise ValueError(
                "Pour Svensson, 'maturities', 'observed_yields' et 'initial_guess' sont requis.")

        def objective(params):
            b0, b1, b2, b3, lam1, lam2 = params
            error = 0.0
            for t, obs in zip(maturities, observed_yields):
                if t == 0:
                    y = b0 + b1 + 0.5 * (b2 + b3)
                else:
                    f1 = t / lam1
                    f2 = t / lam2
                    t1 = (1 - np.exp(-f1)) / f1
                    t2 = t1 - np.exp(-f1)
                    t3 = (1 - np.exp(-f2)) / f2
                    y = b0 + b1 * t1 + b2 * t2 + b3 * t3
                error += (y - obs) ** 2
            return error

        res = minimize(objective, initial_guess, method='Nelder-Mead')
        b0, b1, b2, b3, lam1, lam2 = res.x
        return cls(b0, b1, b2, b3, lam1, lam2)

    def plot_fit(self,
                 maturities: np.ndarray,
                 observed_yields: np.ndarray,
                 n_points: int = 100,
                 title: str = "Calibration du modèle Svensson") -> None:
        """
        Trace la courbe calibrée et les données observées.
        """
        x_fine = np.linspace(min(maturities), max(maturities), n_points)
        y_fit = np.array([self.yield_value(t) for t in x_fine])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=maturities,
                                 y=observed_yields,
                                 mode='markers',
                                 name='Observé',
                                 marker=dict(color='black')))
        fig.add_trace(go.Scatter(x=x_fine,
                                 y=y_fit,
                                 mode='lines',
                                 name='Svensson'))
        fig.update_layout(title=title,
                          xaxis_title='Échéance (années)',
                          yaxis_title='Taux (%)',
                          template='plotly_white')
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
