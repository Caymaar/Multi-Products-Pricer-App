import numpy as np
from typing import Union
from .abstract_stochastic import AbstractStochasticProcess
from market.market import Market

class GBMProcess(AbstractStochasticProcess):
    # ---------------- GBMProcess Class ----------------
    def __init__(self, market : Market, dt : float , n_paths : int, n_steps: int, t_div: float = None, rate_model=None, compute_antithetic=False, seed=None):
        super().__init__(n_paths, n_steps, dt, seed)
        self.market = market
        self.paths_scalar = None
        self.paths_vectorized = None
        self.t_div = t_div
        self.rate_model = rate_model
        self.compute_antithetic = compute_antithetic

    def _apply_dividends(self, paths: np.ndarray):
        """
        Applique les dividendes discrets.
        """
        if self.market.div_type == "discrete" and self.t_div is not None:
            div_reduction_factor = 1 - (self.market.dividend / paths[:, self.t_div])
            paths[:, self.t_div + 1:] *= div_reduction_factor[:, np.newaxis]

    def get_factors(self, dW: np.ndarray, r: Union[float, np.ndarray] = None) -> np.ndarray:
        """
        Calcule les facteurs multiplicatifs de la dynamique GBM, avec support du taux
        constant ou du taux dynamique (par trajectoire et dans le temps).

        :param dW: Mouvement brownien simulé (n_paths, n_steps)
        :param r: Taux d'intérêt (scalaire ou array (n_paths, n_steps))
        :return: Facteurs multiplicatifs cumulés (n_paths, n_steps)
        """
        if r is None:
            # time grid t1, t2, …, t_n
            times = np.arange(1, self.n_steps + 1) * self.dt  # shape (n_steps,)
            # yield_value vectorisée sur le vecteur times → shape (n_steps,)
            r_vals = np.vectorize(self.market.zero_rate)(times) #if callable(self.market.zero_rate) else self.market.zero_rate
            # on broadcast pour obtenir (n_paths, n_steps)
            r = np.broadcast_to(r_vals, dW.shape)

        # Dividend
        q = self.market.dividend if self.market.div_type == "continuous" else 0

        # Drift + diffusion (vectorisé pour taux constants ou dynamiques)
        drift = (r - q - 0.5 * self.market.sigma ** 2) * self.dt
        diffusion = self.market.sigma * dW

        increments = drift + diffusion  # (n_paths, n_steps)

        # Cumul des log-returns
        np.cumsum(increments, axis=1, out=increments)

        return np.exp(increments)  # retourne S_t / S_0 à chaque date

    def simulate(self) -> np.ndarray:
        """
        Simule les trajectoires du sous-jacent
        t avec ou sans taux dynamique (Vasicek),
        en utilisant des browniens corrélés si nécessaire.
        """
        if self.paths_vectorized is None:
            paths = np.empty((self.n_paths, self.n_steps + 1))
            paths[:, 0] = self.market.S0

            # Cas simple : taux constant
            if self.rate_model is None:
                if self.compute_antithetic and self.n_paths % 2 == 0:
                    dW = self.brownian.vectorized_motion(self.n_paths // 2, self.n_steps)
                    dW = np.concatenate((dW, -dW), axis=0)
                else:
                    dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)

                # Calcul des incréments pour tous les chemins
                increments = self.get_factors(dW)

            else:
                # Cas taux dynamique : on a besoin de browniens corrélés
                if self.market.corr_matrix is None:
                    raise ValueError("Corrélation requise pour taux dynamique.")

                dW_corr = self.brownian.vectorized_motion(self.n_paths, self.n_steps, self.market.corr_matrix)
                dW_S = dW_corr[:, :, 0]  # brownien pour sous-jacent
                dW_r = dW_corr[:, :, 1]  # brownien pour taux

                # Simulation des taux dynamiques avec brownien fourni
                rates = self.rate_model.simulate(dW=dW_r)  # shape (n_paths, n_steps + 1)

                # Calcul des facteurs multiplicatifs avec taux dynamiques
                increments = self.get_factors(dW_S, r=rates[:, :-1])  # on aligne r et dW sur les steps

            # Composition des trajectoires du sous-jacent
            paths[:, 1:] = self.market.S0 * increments

            # Application des dividendes discrets (si nécessaire)
            self._apply_dividends(paths)

            self.paths_vectorized = paths

        return self.paths_vectorized