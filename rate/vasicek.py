import numpy as np
from rate.abstract_taux import AbstractRateModel
import matplotlib.pyplot as plt
from stochastic_process.ou_process import OUProcess
from typing import Union
import pandas as pd
import os

class VasicekModel(AbstractRateModel, OUProcess):
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

    @staticmethod
    def calibrate_from_file(source: str = "sofr", **kwargs) -> 'VasicekModel':
        """
        Calibre les paramètres à partir d'un fichier CSV (SOFR, LIBOR...).
        """
        DATA_PATH = os.path.dirname(__file__) + "/../data_taux"
        source = source.lower()
        file_map = {
            "sofr": ("sofr_data.csv", "SOFR (O/N)"),
            "libor": ("libor.csv", "LIBOR 3M ICE")
        }

        if source not in file_map:
            raise ValueError("Source inconnue")

        file_name, column_name = file_map[source]
        df =  pd.read_csv(f"{DATA_PATH}/{file_name}", parse_dates=["DATES"], dayfirst=True)
        df.sort_values("DATES", inplace=True)
        df.reset_index(drop=True, inplace=True)
        r = df[column_name].dropna().values / 100
        dates = df["DATES"].dropna().values

        # Calcul dynamique de `dt` à partir des différences entre les dates
        date_deltas = np.diff(dates).astype('timedelta64[D]')  # Différences entre les dates en jours
        average_delta_days = np.mean(date_deltas).astype(float)  # Pas moyen en jours
        dt = average_delta_days / 365.0  # Conversion en années

        # Calcul de `n_steps` : basé sur le nombre d'observations
        n_steps = len(r) - 1  # Le nombre de pas correspond aux différences entre les observations

        # Vérification que les taux et les pas de temps sont cohérents
        if len(r) < 2 or np.isnan(dt):
            raise ValueError("Les données fournies ne permettent pas un calcul cohérent des paramètres du modèle.")

        # Appel à la méthode de classe pour calibration
        return VasicekModel.calibrate(observed_yields=r, dt=dt, n_steps=n_steps, **kwargs)

if __name__ == "__main__":
    model = VasicekModel.calibrate_from_file("sofr")
    print(model.theta, model.mu, model.sigma)

    T = 5.0
    n_paths = 10

    paths = model.simulate()
    time_grid = np.linspace(0, T, paths.shape[1])

    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(time_grid, paths[i])
    plt.title("Simulation du modèle Vasicek")
    plt.xlabel("Temps (années)")
    plt.ylabel("Taux court")
    plt.axhline(model.mu, color='r', ls='--', label='Taux long terme (b)')
    plt.legend()
    plt.grid()
    plt.show()