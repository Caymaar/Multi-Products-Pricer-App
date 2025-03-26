import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils.manage_bloomberg_data import OptionDataParser


class OptionVolSurface:
    def __init__(self, data):
        """
        Initialise la surface de volatilité à partir d'un DataFrame.
        Le DataFrame doit contenir les colonnes : 'strike', 'maturity', 'vol'

        :param data: pd.DataFrame contenant les données d'options
        """
        self.data = data.copy()
        # Récupération des valeurs uniques de strikes et maturités
        self.strikes = np.sort(self.data['strike'].unique())
        self.maturities = np.sort(self.data['maturity'].unique())
        # Création de la matrice de volatilité : lignes = maturités, colonnes = strikes
        self.vol_matrix = self._create_vol_matrix()

    def _create_vol_matrix(self):
        """
        Construit une matrice de volatilité à partir des données brutes.
        En cas de multiples valeurs pour une même (maturité, strike), la moyenne est prise.
        Si une valeur manque, une interpolation linéaire sur l'axe strike est réalisée.
        """
        vol_matrix = np.zeros((len(self.maturities), len(self.strikes)))
        for i, m in enumerate(self.maturities):
            for j, k in enumerate(self.strikes):
                vols = self.data[(self.data['maturity'] == m) & (self.data['strike'] == k)]['vol']
                if not vols.empty:
                    vol_matrix[i, j] = vols.mean()
                else:
                    vol_matrix[i, j] = np.nan
            # Remplissage des valeurs manquantes par interpolation sur l'axe strike
            row = vol_matrix[i, :]
            if np.isnan(row).any():
                valid = ~np.isnan(row)
                if valid.sum() >= 2:
                    f = interp1d(self.strikes[valid], row[valid], bounds_error=False, fill_value="extrapolate")
                    vol_matrix[i, :] = f(self.strikes)
                else:
                    # Si peu de données sont disponibles, on peut les remplir par la moyenne de la ligne
                    vol_matrix[i, :] = np.nanmean(row)
        return vol_matrix

    def get_vol_smile(self, maturity):
        """
        Retourne le smile (les volatilités en fonction du strike) pour une maturité donnée.
        Si la maturité n'est pas présente dans les données, une interpolation sur l'axe maturité est effectuée.

        :param maturity: maturité recherchée (doit être cohérente avec l'unité utilisée dans les données)
        :return: (strikes, vols) : strikes et volatilités associées
        """
        if maturity in self.maturities:
            index = np.where(self.maturities == maturity)[0][0]
            return self.strikes, self.vol_matrix[index, :]
        else:
            # Interpolation linéaire sur l'axe maturité pour chaque strike
            vols_interpolated = []
            for j, k in enumerate(self.strikes):
                f = interp1d(self.maturities, self.vol_matrix[:, j], bounds_error=False, fill_value="extrapolate")
                vols_interpolated.append(f(maturity))
            return self.strikes, np.array(vols_interpolated)

    def plot_vol_smile(self, maturity):
        """
        Trace le smile (volatilité en fonction du strike) pour une maturité donnée.
        """
        strikes, vols = self.get_vol_smile(maturity)
        plt.figure(figsize=(8, 5))
        plt.plot(strikes, vols, marker='o', linestyle='-')
        plt.xlabel('Strike')
        plt.ylabel('Volatilité Implicite')
        plt.title(f'Volatility Smile pour maturité {maturity}')
        plt.grid(True)
        plt.show()

    def plot_vol_surface(self):
        """
        Trace la surface de volatilité en 3D.
        """
        X, Y = np.meshgrid(self.strikes, self.maturities)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.vol_matrix, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturité')
        ax.set_zlabel('Volatilité Implicite')
        ax.set_title('Surface de Volatilité')
        plt.show()


# Exemple d'utilisation (test) si le module est exécuté en tant que script principal
if __name__ == "__main__":
    file_path = "../data_options/options_data_TSLA 2.xlsx"  # Adapté au fichier pour AAPL
    df_options = OptionDataParser.prepare_option_data(file_path)
    print("Données extraites :", df_options)

    # Création de la surface de volatilité
    vol_surface = OptionVolSurface(df_options)

    # Choix d'une maturité pour afficher le smile : ici, on prend la maturité médiane
    sorted_maturities = vol_surface.maturities
    chosen_maturity = sorted_maturities[len(sorted_maturities) // 2]
    print(f"Maturité choisie pour le smile : {chosen_maturity:.2f} an(s)")

    # Affichage du smile de volatilité pour la maturité choisie
    vol_surface.plot_vol_smile(chosen_maturity)

    # Affichage de la surface complète de volatilité
    vol_surface.plot_vol_surface()
