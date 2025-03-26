import numpy as np
from .abstract_stochastic import AbstractStochasticProcess


class GeometricBrownianMotionProcess(AbstractStochasticProcess):
    def __init__(self, initialValue, mu, sigma):
        """
        Initialise le processus de mouvement brownien géométrique.

        :param initialValue: Valeur initiale S0
        :param mu: Taux de drift (espérance)
        :param sigma: Volatilité
        """
        self.initialValue = initialValue
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T, n_paths, n_steps):
        """
        Simule le processus géométrique sur l'horizon T.

        La trajectoire est donnée par:
        S(t) = S0 * exp[(mu - 0.5*sigma^2)*t + sigma*W(t)]
        où W(t) est un mouvement brownien.
        """
        dt = T / n_steps
        # Incréments du mouvement brownien
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(n_steps, n_paths))
        # Calcul cumulatif pour obtenir W(t)
        W = np.cumsum(dW, axis=0)
        time_grid = np.linspace(dt, T, n_steps)
        drift = (self.mu - 0.5 * self.sigma ** 2) * time_grid[:, np.newaxis]
        diffusion = self.sigma * W
        paths = self.initialValue * np.exp(drift + diffusion)
        return time_grid, paths