import numpy as np
from scipy.interpolate import interp1d

from rate.abstract_taux import AbstractYieldCurve

class RateInterpolation(AbstractYieldCurve):
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

    def calibrate(self):
        pass # Pas de calibration ici

if __name__ == "__main__":

    from data.management.data_retriever import DataRetriever
    from rate.zc_curve import ZeroCouponCurveBuilder
    from datetime import datetime
    from data.management.data_utils import tenor_to_years
    from matplotlib import pyplot as plt
    import pandas as pd

    np.random.seed(272)

    valuation_date = datetime(2023, 10, 1)
    dr = DataRetriever("AMAZON")
    curve_spot = dr.get_risk_free_curve(valuation_date) / 100  # en décimal

    # convertir les tenors (ex. "1Y", "6M") en années
    maturities = np.array([tenor_to_years(t) for t in curve_spot.index])
    zcb = ZeroCouponCurveBuilder(maturities, curve_spot.values)
    zero_rates = zcb.zero_rates

    # interpolation raffinée (cubic via scipy)
    interp = RateInterpolation(maturities, zero_rates, kind="refined cubic")
    interp_y = interp.yield_curve_array(maturities)

    # === 2) Afficher les données brutes dans la console ===
    df_raw = pd.DataFrame({
        "Maturité (années)": maturities,
        "Taux zéro brut": zero_rates
    })
    print("\nDonnées brutes de taux zéro-coupon :")
    print(df_raw.to_string(index=False))

    # === 3) Graphique re-sublimé ===
    plt.figure(figsize=(10, 6))

    # points bruts
    plt.scatter(maturities,
                zero_rates,
                marker="s",
                label="Taux zéro brut",
                zorder=5)

    # courbe interpolée
    plt.plot(maturities,
             interp_y,
             linewidth=2,
             label="Interpolation raffinée (cubic)")

    # annotation de chaque point brut
    for x, y in zip(maturities, zero_rates):
        plt.text(x,
                 y,
                 f"{y * 100:.2f} %",
                 ha="center",
                 va="bottom",
                 fontsize=8)

    plt.title("Courbe Zéro-Coupon : données brutes vs interpolation")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux Zéro-Coupon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()