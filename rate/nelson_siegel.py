import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from rate.abstract_taux import AbstractYieldCurve


class NelsonSiegelModel(AbstractYieldCurve):
    def __init__(self, beta0: float, beta1: float, beta2: float, lambda1: float):
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

    def yield_value(self, t: float) -> float:
        """
        Calcule le taux pour une échéance donnée t selon Nelson-Siegel.
        """
        if t == 0:
            # Limite pour éviter la division par zéro
            return self.beta0 + self.beta1 + 0.5 * self.beta2

        factor = t / self.lambda1
        term1 = (1 - np.exp(-factor)) / factor
        term2 = term1 - np.exp(-factor) - np.exp(-factor)
        return self.beta0 + self.beta1 * term1 + self.beta2 * term2

    @classmethod
    def calibrate(cls,
                  maturities: np.ndarray,
                  observed_yields: np.ndarray,
                  initial_guess: np.ndarray) -> 'NelsonSiegelModel':
        """
        Calibre les paramètres du modèle Nelson-Siegel et renvoie une instance configurée.

        :param maturities: tableau des échéances (en années)
        :param observed_yields: tableau des rendements observés
        :param initial_guess: estimation initiale des paramètres [beta0, beta1, beta2, lambda1]
        :return: instance NelsonSiegelModel calibrée
        """
        if maturities is None or observed_yields is None or initial_guess is None:
            raise ValueError(
                "Pour Nelson-Siegel, 'maturities', 'observed_yields' et 'initial_guess' sont requis.")

        def objective(params):
            b0, b1, b2, lam = params
            errors = []
            for t, obs in zip(maturities, observed_yields):
                if t == 0:
                    y = b0 + b1 + 0.5 * b2
                else:
                    fac = t / lam
                    t1 = (1 - np.exp(-fac)) / fac
                    t2 = t1 - np.exp(-fac) - np.exp(-fac)
                    y = b0 + b1 * t1 + b2 * t2
                errors.append((y - obs) ** 2)
            return np.sum(errors)

        res = minimize(objective, initial_guess, method='Nelder-Mead')
        beta0, beta1, beta2, lambda1 = res.x
        return cls(beta0, beta1, beta2, lambda1)

    def plot_fit(self,
                 maturities: np.ndarray,
                 observed_yields: np.ndarray,
                 n_points: int = 100,
                 title: str = "Calibration du modèle Nelson-Siegel") -> None:
        """
        Trace la courbe calibrée et les données observées.
        """
        maturities_fine = np.linspace(min(maturities), max(maturities), n_points)
        fitted = np.array([self.yield_value(t) for t in maturities_fine])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=maturities, y=observed_yields,
                                 mode='markers', name='Observé', marker=dict(color='black')))
        fig.add_trace(go.Scatter(x=maturities_fine, y=fitted,
                                 mode='lines', name='Nelson-Siegel'))
        fig.update_layout(title=title,
                          xaxis_title='Échéance (années)',
                          yaxis_title='Taux (%)',
                          template='plotly_white')
        fig.show()


if __name__ == "__main__":

    from data.management.data_retriever import DataRetriever
    from rate.zc_curve import ZeroCouponCurveBuilder
    from datetime import datetime
    from data.management.data_utils import tenor_to_years

    np.random.seed(272)

    DR = DataRetriever("LVMH")

    date = datetime(year=2023,month=10,day=1)
    curve = DR.get_risk_free_curve(date)
    spot = DR.get_risk_free_index(date)
    maturity = np.array([tenor_to_years(tenor=t) for t in curve.index])

    zc = ZeroCouponCurveBuilder(maturity,curve.values)

    # --- Calibration des modèles ---
    initial_guess_ns = [0.02, -0.01, 0.01, 1.5]
    params_ns = NelsonSiegelModel.calibrate(maturity, zc.zero_rates, np.array(initial_guess_ns))

    print("\nParamètres calibrés (Nelson-Siegel) :")
    print(
        f"beta0 = {params_ns.beta0:.4f}, beta1 = {params_ns.beta1:.4f}, beta2 = {params_ns.beta2:.4f}, lambda1 = {params_ns.lambda1:.4f}")

    params_ns.plot_fit(maturity, zc.zero_rates, title="Calibration du modèle Nelson-Siegel")