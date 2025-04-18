from abc import ABC, abstractmethod
import numpy as np

# ---------------- Courbe des taux ----------------
class AbstractRateModel(ABC):
    @abstractmethod
    def yield_value(self, maturity):
        """
        Calcule le taux pour une échéance donnée tau.

        :param maturity: float, l'échéance pour laquelle on souhaite calculer le taux
        :return: le taux pour l'échéance donnée
        """
        pass

    def yield_curve_array(self, maturities):
        """
        Calcule la courbe des taux pour un tableau d'échéances.

        :param maturities: array-like, les échéances pour lesquelles on souhaite calculer les taux
        :return: array-like, les taux correspondants
        """
        return np.array([self.yield_value(t) for t in maturities])

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