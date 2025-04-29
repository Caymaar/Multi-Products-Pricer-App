from abc import ABC, abstractmethod
import numpy as np

# ---------------- Courbe des taux ----------------
class AbstractYieldCurve(ABC):
    @abstractmethod
    def yield_value(self, maturity) -> float:
        """
        Calcule le taux pour une échéance donnée tau.

        :param maturity: float, l'échéance pour laquelle on souhaite calculer le taux
        :return: le taux pour l'échéance donnée
        """
        pass

    def yield_curve_array(self, maturities: np.ndarray) -> np.ndarray:
        """
        Calcule la courbe des taux pour un tableau d'échéances.

        :param maturities: array-like, les échéances pour lesquelles on souhaite calculer les taux
        :return: array-like, les taux correspondants
        """
        return np.array([self.yield_value(t) for t in maturities])

    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward discret basé sur les taux zéro."""
        z1 = self.yield_value(t1)
        z2 = self.yield_value(t2)
        return (z2 * t2 - z1 * t1) / (t2 - t1)

    def instantaneous_forward(self, t: float, h: float = 1e-4) -> float:
        """Forward instantané par dérivation centrale."""
        z1 = self.yield_value(t - h)
        z2 = self.yield_value(t + h)
        dzdt = (z2 - z1) / (2 * h)
        return self.yield_value(t) + t * dzdt

    def discount_factor(self, t: float) -> float:
        return np.exp(-self.yield_value(t) * t)

    def forward_from_zc_ratio(self, t1: float, t2: float) -> float:
        """Méthode alternative basée sur les facteurs d'actualisation (ton approche)."""
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (df1 / df2 - 1) / (t2 - t1)

    @abstractmethod
    def calibrate(self, **kwargs):
        """
        Calibre les paramètres du modèle en minimisant l'erreur quadratique entre
        les rendements observés et ceux générés par le modèle.

        :param maturities: array-like, les échéances
        :param observed_yields: array-like, les rendements observés
        :param initial_guess: array-like, estimation initiale des paramètres
        :param dt: valeur du pas de temps pour simulations
        :param n_steps: nombre de pas de temps pour simulations
        :return: array-like, les paramètres calibrés
        """
        pass

