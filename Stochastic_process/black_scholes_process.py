import numpy as np
from .abstract_stochastic import AbstractStochasticProcess


class BlackScholesProcess(AbstractStochasticProcess):
    def __init__(self, S0, r, sigma):
        """
        Initialise le processus Black-Scholes.

        :param S0: Valeur initiale de l'actif
        :param r: Taux sans risque (drift)
        :param sigma: Volatilité de l'actif
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma

    def simulate(self, T, n_paths, n_steps):
        """
        Simule le processus Black-Scholes sur l'horizon T.

        La trajectoire suit la formule :
            S(t) = S0 * exp[(r - 0.5*sigma^2)*t + sigma*W(t)]
        où W(t) est un mouvement brownien.

        :param T: Temps total de la simulation
        :param n_paths: Nombre de trajectoires à simuler
        :param n_steps: Nombre de pas de temps
        :return: Tuple (time_grid, paths) contenant la grille de temps et les trajectoires simulées
        """
        dt = T / n_steps
        # Incréments du mouvement brownien
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(n_steps, n_paths))
        # Calcul cumulatif pour obtenir W(t)
        W = np.cumsum(dW, axis=0)
        time_grid = np.linspace(dt, T, n_steps)
        drift = (self.r - 0.5 * self.sigma ** 2) * time_grid[:, np.newaxis]
        diffusion = self.sigma * W
        paths = self.S0 * np.exp(drift + diffusion)
        return time_grid, paths
