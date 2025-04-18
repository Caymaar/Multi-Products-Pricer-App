import numpy as np
from .abstract_stochastic import AbstractStochasticProcess

class OUProcess(AbstractStochasticProcess):
    """Processus d'Ornstein-Uhlenbeck."""

    def __init__(self, theta: float, mu: float, sigma: float, initial_value: float,
                 n_paths: int, n_steps: int, dt: float, compute_antithetic=False, seed: int = None):
        """
        Initialise le processus OU.

        :param theta: Vitesse de réversion
        :param mu: Moyenne de long terme
        :param sigma: Volatilité
        :param initial_value: Valeur initiale X0
        :param n_paths: Nombre de trajectoires
        :param n_steps: Nombre de pas de temps
        :param dt: Pas de temps
        :param seed: Seed pour la reproductibilité
        """
        super().__init__(n_paths, n_steps, dt, seed)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.initial_value = initial_value
        self.compute_antithetic = compute_antithetic

    def simulate_single_path(self):
        """
        Simule une trajectoire unique du processus d'Ornstein-Uhlenbeck avec motion scalaire.

        :return: Numpy array contenant la trajectoire simulée.
        """
        path = np.empty(self.n_steps + 1)
        path[0] = self.initial_value  # Valeur initiale du processus

        for t in range(1, self.n_steps + 1):
            dW = self.brownian.scalar_motion() * np.sqrt(self.dt)  # Incrément brownien
            path[t] = (path[t - 1] +
                       self.theta * (self.mu - path[t - 1]) * self.dt +
                       self.sigma * dW)

        return path

    def simulate_step(self, r_t_previous: float, dW_t: float, dt: float) -> float:
        """
        Calcule un taux r_t à l'instant suivant selon le processus d'Ornstein-Uhlenbeck.

        :param r_t_previous: Taux à l'instant précédent (valeur scalaire).
        :param dW_t: Incrément Brownien pour le pas de temps actuel.
        :param dt: Taille du pas de temps.
        :return: Taux simulé pour l'instant suivant.
        """
        return r_t_previous + self.theta * (self.mu - r_t_previous) * dt + self.sigma * np.sqrt(dt) * dW_t

    def simulate(self) -> np.ndarray:
        """Simule des trajectoires de OU."""
        paths = np.empty((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.initial_value

        if self.compute_antithetic and self.n_paths % 2 == 0:
            dW = self.brownian.vectorized_motion(self.n_paths // 2, self.n_steps)
            dW = np.concatenate((dW, -dW), axis=0)
        else:
            dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)

        exp_factor = np.exp(-self.theta * self.dt)

        for t in range(1, self.n_steps + 1):
            paths[:, t] = (paths[:, t - 1] * exp_factor +
                           self.mu * (1 - exp_factor) +
                           self.sigma * np.sqrt((1 - exp_factor**2) / (2 * self.theta)) * dW[:, t - 1])

        return paths
