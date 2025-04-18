import numpy as np
from .abstract_stochastic import AbstractStochasticProcess

class GBMProcess(AbstractStochasticProcess):
    # ---------------- GBMProcess Class ----------------
    def __init__(self, market, dt, n_paths, n_steps, t_div, rate_model=None, correlation_matrix=None, compute_antithetic=False, seed=None):
        super().__init__(n_paths, n_steps, dt, seed)
        self.market = market
        self.paths_scalar = None
        self.paths_vectorized = None
        self.t_div = t_div
        self.rate_model = rate_model
        self.correlation_matrix = correlation_matrix
        self.compute_antithetic = compute_antithetic

    def _apply_dividends(self, paths: np.ndarray):
        """
        Applique les dividendes discrets.
        """
        if self.market.div_type == "discrete" and self.t_div is not None:
            div_reduction_factor = 1 - (self.market.dividend / paths[:, self.t_div])
            paths[:, self.t_div + 1:] *= div_reduction_factor[:, np.newaxis]

    def get_factors(self, dW: float, r = None) -> float:
        """
        Calcul de l'incrément selon la formule : exp((r - q - 0.5 * sigma^2) * dt + sigma * dW)
        Si dW est un scalaire, effectue le calcul pour une seule trajectoire.
        Si dW est une matrice, effectue le calcul pour toutes les trajectoires.
        """
        if r is None:
            r = self.market.r

        growth_adjustment = (r - (self.market.dividend if self.market.div_type == "continuous" else 0)
                             - 0.5 * self.market.sigma ** 2) * self.dt

        increments = growth_adjustment + self.market.sigma * dW
        # Effectuer le cumul des incréments si dW est un tableau
        #np.cumsum(increments, axis=1, out=increments)
        return np.exp(increments)

    def simulate3(self, batch_size=100):
        """
        Simule les trajectoires du sous-jacent avec taux dynamiques et Brownien antithétique.

        :param batch_size: Taille des blocs pour limiter la mémoire utilisée.
        :return: Un tableau numpy contenant les trajectoires finales (n_paths,).
        """
        global dW_S, dW_r

        final_paths = np.empty((self.n_paths,self.n_steps + 1), dtype=np.float32)  # Stocke uniquement les valeurs finales

        for i in range(0, self.n_paths, batch_size):
            current_batch = min(batch_size, self.n_paths - i)

            # Initialisation des trajectoires pour le batch
            paths = np.empty((current_batch, self.n_steps + 1), dtype=np.float32)
            paths[:,0] = self.market.S0

            # Génération des mouvements Brownien avec antithétique
            dW = self.brownian.vectorized_motion(current_batch, self.n_steps)
            if dW.ndim == 3:
                dW_S = dW[:, :, 0]  # Brownien pour le sous-jacent
                dW_r = dW[:, :, 1]  # Brownien pour les taux


            # Initialisation des taux : dynamiques (rate_model) ou constants (self.market.r)
            if self.rate_model is not None:
                rates = np.full(current_batch, self.rate_model.initial_value, dtype=np.float32)  # Taux initiaux depuis rate_model
            else:
                rates = np.full(current_batch, self.market.r, dtype=np.float32)  # Taux constant de marché

            for t in range(self.n_steps):
                # Simulation des taux dynamiques
                if self.rate_model is not None and dW.ndim == 3:
                    rates = self.rate_model.simulate_step(r_t_previous=rates, dW_t=dW_r[:, t], dt=self.dt)

                # Calcul des facteurs avec get_factors
                increments = self.get_factors(r=rates, dW=dW_S[:, t]) if dW.ndim == 3 else self.get_factors(r=rates, dW=dW[:, t])

                # Mise à jour des trajectoires
                paths[:, t + 1] = paths[:, t] * increments

            # Sauvegarde des valeurs finales
            final_paths[i:i + current_batch] = paths

        return final_paths

    def simulate(self):
        """
        Simule toutes les trajectoires du sous-jacent en une seule opération vectorielle.
        Optimisation complète sans boucles temporelles.

        :return: Un tableau numpy contenant toutes les trajectoires (n_paths, n_steps + 1).
        """
        global dW_S, dW_r

        # 1. Initialisation des trajectoires : (n_paths, n_steps + 1)
        paths = np.empty((self.n_paths, self.n_steps + 1), dtype=np.float32)
        paths[:, 0] = self.market.S0  # Valeur initiale du sous-jacent

        # 2. Initialisation des taux
        if self.rate_model is not None:
            rates = np.full(self.n_paths, self.rate_model.initial_value, dtype=np.float32)  # Taux initiaux du modèle
        else:
            rates = np.full(self.n_paths, self.market.r, dtype=np.float32)  # Taux constants

        # 3. Génération des mouvements Brownien vectorisés
        if self.correlation_matrix:
            dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps, self.market.correlation)
        else:
            dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)

        if dW.ndim == 3:
            dW_S = dW[:, :, 0]  # Brownien pour le sous-jacent
            dW_r = dW[:, :, 1]  # Brownien pour les taux
        else:
            dW_S = dW
            dW_r = None

        # 4. Calcul des trajectoires via simulate_step
        for t in range(self.n_steps):
            # Mise à jour des taux dynamiques si le modèle de taux est défini
            if self.rate_model is not None and dW_r is not None:
                rates = self.rate_model.simulate_step(r_t_previous=rates, dW_t=dW_r[:, t], dt=self.dt)

            # Calcul des facteurs avec get_factors
            increments = self.get_factors(r=rates, dW=dW_S[:, t])

            # Mise à jour des trajectoires du sous-jacent
            paths[:, t + 1] = paths[:, t] * increments
        return paths


    def simulate2(self):
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
