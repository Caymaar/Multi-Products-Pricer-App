import numpy as np
from .abstract_stochastic import AbstractStochasticProcess

class GBMProcess(AbstractStochasticProcess):
    # ---------------- GBMProcess Class ----------------
    def __init__(self, market, dt, n_paths, n_steps, t_div, compute_antithetic=False, seed=None):
        super().__init__(n_paths, n_steps, dt, seed)
        self.market = market
        self.paths_scalar = None
        self.paths_vectorized = None
        self.t_div = t_div
        self.compute_antithetic = compute_antithetic

    def get_factors(self, dW: float) -> float:
        """
        Calcul de l'incrément selon la formule : exp((r - q - 0.5 * sigma^2) * dt + sigma * dW)
        Si dW est un scalaire, effectue le calcul pour une seule trajectoire.
        Si dW est une matrice, effectue le calcul pour toutes les trajectoires.
        """
        growth_adjustment = (self.market.r - (self.market.dividend if self.market.div_type == "continuous" else 0)
                             - 0.5 * self.market.sigma ** 2) * self.dt

        if isinstance(dW, float):
            return np.exp(growth_adjustment + self.market.sigma * dW)

        # Si dW est une matrice (pour la méthode vectorielle)
        elif isinstance(dW, np.ndarray):
            increments = growth_adjustment + self.market.sigma * dW
            # Effectuer le cumul des incréments si dW est un tableau
            np.cumsum(increments, axis=1, out=increments)
            return np.exp(increments)

    def _apply_dividends(self, paths: np.ndarray):
        """
        Applique les dividendes discrets.
        """
        if self.market.div_type == "discrete" and self.t_div is not None:
            div_reduction_factor = 1 - (self.market.dividend / paths[:, self.t_div])
            paths[:, self.t_div + 1:] *= div_reduction_factor[:, np.newaxis]

    def simulate(self):
        """
        Génère des trajectoires GBM avec une méthode vectorielle (sans boucles).
        """
        if self.paths_vectorized is None:
            # Initialisation des chemins
            paths = np.empty((self.n_paths, self.n_steps + 1))
            paths[:, 0] = self.market.S0  # Initialisation à S0

            if self.compute_antithetic and self.n_paths % 2 == 0:
                dW = self.brownian.vectorized_motion(self.n_paths // 2, self.n_steps)
                dW = np.concatenate((dW, -dW), axis=0)
            else:
                dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)

            # Calcul des incréments pour tous les chemins
            increments = self.get_factors(dW)

            # Diffusions
            paths[:, 1:] = self.market.S0 * increments

            # Application des dividendes discrets éventuels
            self._apply_dividends(paths)

            self.paths_vectorized = paths

        return self.paths_vectorized
