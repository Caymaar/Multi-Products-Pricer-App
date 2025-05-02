from rate.abstract_taux import AbstractYieldCurve
from stochastic_process.ou_process import OUProcess
from typing import Union
import numpy as np


class VasicekModel(AbstractYieldCurve, OUProcess):
    """
    Modèle de taux court de Vasicek basé sur le processus d'Ornstein-Uhlenbeck.
    Hérite de OUProcess pour intégrer les méthodes de simulation.
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float, dt: float, n_steps: int, n_paths: int = 10000, compute_antithetic: bool = False, seed: int = None):
        """
        Initialise le modèle Vasicek en utilisant les caractéristiques du processus OU.

        :param a: Vitesse de réversion
        :param b: Taux d'intérêt de long terme
        :param sigma: Volatilité
        :param r0: Taux initial
        :param n_paths: Nombre de trajectoires (par défaut : 1000).
        :param n_steps: Nombre de pas de temps (par défaut : 252).
        :param dt: Pas de temps (par défaut : 1/252).
        :param compute_antithetic: Activer le sampling antithétique (par défaut : False).
        :param seed: Seed pour la reproductibilité.
        """
        OUProcess.__init__(self, theta=a, mu=b, sigma=sigma, initial_value=r0, n_paths=n_paths, n_steps=n_steps, dt=dt, compute_antithetic=compute_antithetic, seed=seed)

    def r(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Espérance du taux court à l'instant t.
        """
        return self.mu + (self.initial_value - self.mu) * np.exp(-self.theta * t)

    def yield_value(self, t: float) -> float:
        """
        Calcule le taux zéro-coupon à maturité t à partir du modèle Vasicek.
        """
        if t == 0:
            return self.initial_value
        B = (1 - np.exp(-self.theta * t)) / self.theta
        A = (
            (B - t) * (self.theta**2 * self.mu - 0.5 * self.sigma**2) / self.theta**2
            - (self.sigma**2 * B**2) / (4 * self.theta)
        )
        P = np.exp(A - B * self.initial_value)
        return -np.log(P) / t

    @classmethod
    def calibrate(cls, observed_yields, dt: float, n_steps: int, **kwargs) -> 'VasicekModel':
        """
        Calibre les paramètres du modèle Vasicek.

        :param observed_yields: Série temporelle des taux
        :param dt: Pas de temps entre les observations (obligatoire)
        :return: Instance calibrée du modèle Vasicek.
        """
        if observed_yields is None or dt is None:
            raise ValueError("Pour le modèle Vasicek, 'observed_yields' et 'dt' doivent être fournis.")

        r_t = observed_yields[:-1]
        r_tp1 = observed_yields[1:]

        X = np.vstack([np.ones(len(r_t)), r_t]).T
        beta, _, _, _ = np.linalg.lstsq(X, r_tp1, rcond=None)
        c, d = beta

        a = (1 - d) / dt
        b = c / (a * dt)

        resid = r_tp1 - (c + d * r_t)
        sigma = np.std(resid, ddof=1) / np.sqrt(dt)

        # Création du modèle directement à partir des paramètres calibrés
        return cls(
            a=a,
            b=b,
            sigma=sigma,
            r0=r_t[0],
            dt=dt,
            n_steps=n_steps,
            **kwargs
        )

if __name__ == "__main__":

    from data.management.data_retriever import DataRetriever
    from zc_curve import ZeroCouponCurveBuilder
    from data.management.data_utils import tenor_to_years
    from datetime import datetime

    DR = DataRetriever("LVMH")

    date = datetime(year=2023,month=10,day=1)
    curve = DR.get_risk_free_curve(date) / 100
    spot = DR.get_risk_free_index(date) /100

    maturity = np.array([tenor_to_years(t) for t in curve.index])
    mat = np.arange(maturity[0], maturity[-1], 0.01)

    zc = ZeroCouponCurveBuilder(maturity, curve.values, freq=1)

    # Tentative peu fructueuse de calibration par interpolation cubique en amont pour obtenir un dt constant à la calibration...
    zc_rates = zc.build_curve(method='interpolation',kind='refined cubic').yield_curve_array(mat)

    VM = VasicekModel.calibrate(zc_rates, 0.01, 1000)
    print(VM.theta, VM.mu, VM.sigma)
