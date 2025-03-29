from abc import ABC, abstractmethod


# ---------------- Interpolation des Taux ----------------
class AbstractTaux(ABC):

    @abstractmethod
    def get_taux(self, x_val):
        """
        Retourne le taux interpolé pour une valeur donnée x_val.

        :param x_val: float ou array-like, la ou les valeurs pour lesquelles interpoler le taux
        :return: le taux interpolé
        """
        pass


# ---------------- Courbe des taux ----------------
class AbstractYieldCurve(ABC):
    @abstractmethod
    def yield_curve(self, tau):
        """
        Calcule le taux pour une échéance donnée tau.

        :param tau: float, l'échéance pour laquelle on souhaite calculer le taux
        :return: le taux pour l'échéance donnée
        """
        pass

    @abstractmethod
    def yield_curve_array(self, taus):
        """
        Calcule la courbe des taux pour un tableau d'échéances.

        :param taus: array-like, les échéances pour lesquelles on souhaite calculer les taux
        :return: array-like, les taux correspondants
        """
        pass

    @abstractmethod
    def calibrate(self, taus, observed_yields, initial_guess):
        """
        Calibre les paramètres du modèle en minimisant l'erreur quadratique entre
        les rendements observés et ceux générés par le modèle.

        :param taus: array-like, les échéances
        :param observed_yields: array-like, les rendements observés
        :param initial_guess: array-like, estimation initiale des paramètres
        :return: array-like, les paramètres calibrés
        """
        pass