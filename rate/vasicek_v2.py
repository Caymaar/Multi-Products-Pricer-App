import numpy as np
import matplotlib.pyplot as plt
from stochastic_process.ou_process import OUProcess
from typing import Union
import pandas as pd
import os

class VasicekModel:
    """
    Modèle de taux court de Vasicek basé sur le processus d'Ornstein-Uhlenbeck.
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        """
        Initialise le modèle de Vasicek.

        :param a: Vitesse de réversion
        :param b: Taux d'intérêt de long terme
        :param sigma: Volatilité
        :param r0: Taux initial
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def r(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Espérance du taux court à l'instant t.
        """
        return self.b + (self.r0 - self.b) * np.exp(-self.a * t)

    def zero_coupon_rate(self, t: float) -> float:
        """
        Calcule le taux zéro-coupon à maturité t à partir du modèle Vasicek.
        """
        if t == 0:
            return self.r0
        B = (1 - np.exp(-self.a * t)) / self.a
        A = (
            (B - t) * (self.a**2 * self.b - 0.5 * self.sigma**2) / self.a**2
            - (self.sigma**2 * B**2) / (4 * self.a)
        )
        P = np.exp(A - B * self.r0)
        return -np.log(P) / t

    def simulate_paths(self, T: float, dt: float, n_paths: int, seed: int = None) -> np.ndarray:
        """
        Simule des trajectoires du taux court selon le modèle Vasicek.
        Utilise le processus d'Ornstein-Uhlenbeck.

        :param T: Horizon de temps (en années)
        :param dt: Pas de temps
        :param n_paths: Nombre de trajectoires
        :param seed: Reproductibilité
        """
        n_steps = int(T / dt)
        ou = OUProcess(
            theta=self.a,
            mu=self.b,
            sigma=self.sigma,
            initial_value=self.r0,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            seed=seed,
        )
        return ou.simulate()

    @staticmethod
    def calibrate_from_time_series(r: np.ndarray, dt: float):
        """
        Calibre les paramètres à partir d'une série temporelle de taux.
        """
        r_t = r[:-1]
        r_tp1 = r[1:]

        X = np.vstack([np.ones(len(r_t)), r_t]).T
        beta, _, _, _ = np.linalg.lstsq(X, r_tp1, rcond=None)
        c, d = beta

        a = (1 - d) / dt
        b = c / (a * dt)

        resid = r_tp1 - (c + d * r_t)
        sigma = np.std(resid, ddof=1) / np.sqrt(dt)

        return VasicekModel(a=a, b=b, sigma=sigma, r0=r[0])

    @staticmethod
    def calibrate_from_file(source: str = "sofr") -> 'VasicekModel':
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
        r = df[column_name].dropna().values

        dt = 1.0 / 252  # marché ouvert ~252 jours/an
        return VasicekModel.calibrate_from_time_series(r, dt)

if __name__ == "__main__":
    model = VasicekModel.calibrate_from_file("sofr")
    print(model.a, model.b, model.sigma)

    T = 5.0
    dt = 1 / 252
    n_paths = 10

    paths = model.simulate_paths(T=T, dt=dt, n_paths=n_paths)
    time_grid = np.linspace(0, T, paths.shape[1])

    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(time_grid, paths[i])
    plt.title("Simulation du modèle Vasicek")
    plt.xlabel("Temps (années)")
    plt.ylabel("Taux court")
    plt.axhline(model.b, color='r', ls='--', label='Taux long terme (b)')
    plt.legend()
    plt.grid()
    plt.show()