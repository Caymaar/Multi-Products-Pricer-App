from abc import ABC, abstractmethod
from .brownian_motion import BrownianMotion

class AbstractStochasticProcess(ABC):
    """Classe mère pour tous les processus stochastiques."""

    def __init__(self, n_paths: int, n_steps: int, dt: float, seed: int = None):
        """
        Initialise un processus stochastique.

        :param n_paths: Nombre de trajectoires simulées
        :param n_steps: Nombre de pas de temps
        :param dt: Pas de temps
        :param seed: Seed pour la reproductibilité des simulations
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        self.brownian = BrownianMotion(dt, seed)  # Instanciation du mouvement brownien

    @abstractmethod
    def simulate(self):
        """Méthode abstraite que chaque classe fille doit implémenter."""
        pass
