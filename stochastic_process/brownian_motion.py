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

    def vectorized_motion(self, n_paths, n_steps) -> np.array :
        self._gen = np.random.default_rng(self.seed) # reset du générateur de nombres aléatoires
        uniform_samples = self._gen.uniform(0, 1, (n_paths, n_steps))
        norm_matrix = stats.norm.ppf(uniform_samples)
        return np.sqrt(self.dt) * norm_matrix