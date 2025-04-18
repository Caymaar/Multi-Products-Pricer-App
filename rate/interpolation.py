import numpy as np
from scipy.interpolate import interp1d

class RateInterpolation:
    def __init__(self, x, y, kind='linear'):
        """
        Initialise l'interpolateur.

        Paramètres:
        - x : array-like, les valeurs de référence (par exemple, temps ou indices)
        - y : array-like, les taux correspondants à chaque valeur de x
        - kind : type d'interpolation ('linear' ou 'cubic')
        """
        if len(x) != len(y):
            raise ValueError("Les tableaux x et y doivent avoir la même longueur")

        self.x = np.array(x)
        self.y = np.array(y)
        self.kind = kind

    def _linear_interpolation(self, x_val):
        """
        Effectue une interpolation linéaire entre deux points.
        """
        # Trouver les indices des deux points voisins
        idx = np.searchsorted(self.x, x_val) - 1
        idx = np.clip(idx, 0, len(self.x) - 2)

        # Calcul de l'interpolation linéaire
        x0, x1 = self.x[idx], self.x[idx + 1]
        y0, y1 = self.y[idx], self.y[idx + 1]

        # Formule de l'interpolation linéaire
        return y0 + (y1 - y0) * (x_val - x0) / (x1 - x0)

    def _cubic_interpolation(self, x_val):
        """
        Effectue une interpolation cubique entre quatre points.
        """
        # Trouver l'intervalle correspondant
        idx = np.searchsorted(self.x, x_val)
        idx = np.clip(idx, 1, len(self.x) - 3)

        # Extraire les points voisins pour l'interpolation cubique
        x0, x1, x2, x3 = self.x[idx - 1: idx + 3]
        y0, y1, y2, y3 = self.y[idx - 1: idx + 3]

        # Résolution du système de spline naturelle :
        dx0 = x1 - x0
        dx1 = x2 - x1
        dx2 = x3 - x2

        a0 = y1
        a1 = (y2 - y0) / dx1 - (y3 - y1) / dx2
        a2 = (y3 - y1) / dx2 - (y2 - y0) / dx1
        a3 = (y3 - 2 * y2 + y1) / (dx1 * dx2)

        # Retourne l'interpolation cubique
        return a0 + a1 * (x_val - x1) + a2 * (x_val - x1) ** 2 + a3 * (x_val - x1) ** 3

    def yield_value(self, x_val):
        """
        Retourne le taux interpolé pour une valeur donnée x_val.
        """
        if self.kind == 'linear':
            return self._linear_interpolation(x_val)
        elif self.kind == 'cubic':
            return self._cubic_interpolation(x_val)
        elif self.kind == 'refined cubic': # utilisation du package pour plus de précision
            interpolator = interp1d(self.x, self.y, kind='cubic', fill_value="extrapolate")
            return float(interpolator(x_val))
        else:
            raise ValueError("Le type d'interpolation doit être 'linear' ou 'cubic'.")

    def yield_curve_array(self, maturities):
        """
        Calcule la courbe des taux interpolés pour un tableau d'échéances.
        """
        return np.array([self.yield_value(t) for t in maturities])