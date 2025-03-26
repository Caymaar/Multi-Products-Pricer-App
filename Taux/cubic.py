import numpy as np
from scipy.interpolate import interp1d
from .abstract_taux import AbstractTaux


class CubicTaux(AbstractTaux):
    def __init__(self, x, y, kind='cubic'):
        """
        Initialise l'interpolateur cubique.

        Paramètres:
        - x : array-like, les valeurs de référence (par exemple, temps ou indices)
        - y : array-like, les taux correspondants à chaque valeur de x
        - kind : type d'interpolation (par défaut 'cubic')
        """
        if len(x) != len(y):
            raise ValueError("Les tableaux x et y doivent avoir la même longueur")

        self.x = np.array(x)
        self.y = np.array(y)
        self.interpolateur = interp1d(self.x, self.y, kind=kind, fill_value="extrapolate")

    def get_taux(self, x_val):
        """
        Retourne le taux interpolé pour une valeur donnée x_val.
        """
        return self.interpolateur(x_val)