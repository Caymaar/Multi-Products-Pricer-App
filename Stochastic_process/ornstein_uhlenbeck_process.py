import numpy as np
from abstract_stochastic import AbstractStochasticProcess


class OrnsteinUhlenbeckProcess(AbstractStochasticProcess):
    def __init__(self, initialValue, theta, mu, sigma):
        """
        Initialise le processus d'Ornstein-Uhlenbeck.

        :param initialValue: Valeur initiale X0
        :param theta: Vitesse de réversion vers la moyenne
        :param mu: Moyenne de long terme
        :param sigma: Volatilité
        """
        self.initialValue = initialValue
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T, n_paths, n_steps):
        """
        Simule le processus d'Ornstein-Uhlenbeck sur l'horizon T.

        La dynamique du processus est donnée par :
            dX(t) = theta * (mu - X(t)) * dt + sigma * dW(t)
        où dW(t) représente l'incrément d'un mouvement brownien.

        :param T: Temps total de la simulation
        :param n_paths: Nombre de trajectoires simulées
        :param n_steps: Nombre de pas de temps
        :return: Tuple (time_grid, paths) contenant la grille de temps et les trajectoires simulées
        """
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = self.initialValue

        for i in range(1, n_steps + 1):
            dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
            paths[i] = paths[i - 1] + self.theta * (self.mu - paths[i - 1]) * dt + self.sigma * dW

        return time_grid, paths
