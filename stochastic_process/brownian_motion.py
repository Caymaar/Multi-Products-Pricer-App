import numpy as np
import scipy.stats as stats

# ---------------- BrownianMotion Class ----------------
class BrownianMotion:
    def __init__(self, dt, seed=None):
        self.seed = seed
        self._gen = np.random.default_rng(seed)
        self.dt = dt

    def scalar_motion(self) -> float :
        p = self._gen.uniform(0, 1)
        while p == 0:
            p = self._gen.uniform(0, 1)
        return stats.norm.ppf(p) * np.sqrt(self.dt)

    def vectorized_motion(self, n_paths, n_steps, correlation_matrix=None):
        """
        Génère des mouvements browniens, avec ou sans corrélation.

        :param n_paths: Nombre de trajectoires.
        :param n_steps: Nombre de pas de temps.
        :param correlation_matrix: Matrice de corrélation (positive définie). Si None, les trajectoires sont indépendantes.
        :return: Matrice des mouvements browniens.
        """
        #self._gen = np.random.default_rng(self.seed)  # Réinitialisation du générateur

        # Générer des échantillons uniformes
        if correlation_matrix is None:
            # Cas sans corrélation : deux dimensions (n_paths, n_steps)
            uniform_samples = self._gen.uniform(0, 1, (n_paths, n_steps))
        else:
            # Cas avec corrélation : trois dimensions (n_paths, n_steps, dimensions de corrélation)
            uniform_samples = self._gen.uniform(0, 1, (n_paths, n_steps, correlation_matrix.shape[0]))

        # Convertir en normal via PPF
        normal_samples = stats.norm.ppf(uniform_samples)

        if correlation_matrix is not None:
            # Appliquer la matrice de corrélation si elle est définie
            L = np.linalg.cholesky(correlation_matrix)
            normal_samples = np.einsum('ij,klj->kli', L, normal_samples)

        return np.sqrt(self.dt) * normal_samples