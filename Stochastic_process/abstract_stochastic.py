from abc import ABC, abstractmethod


class AbstractStochasticProcess(ABC):
    @abstractmethod
    def simulate(self, T, n_paths, n_steps):
        """
        Simule le processus stochastique sur un horizon T avec n_steps pas de temps pour n_paths trajectoires.

        :param T: Horizon total (temps)
        :param n_paths: Nombre de trajectoires à simuler
        :param n_steps: Nombre de pas de temps
        :return: (time_grid, paths)
                 - time_grid: tableau des temps
                 - paths: tableau des trajectoires simulées (dimension: n_steps x n_paths)
        """
        pass