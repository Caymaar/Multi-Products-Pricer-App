import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd


# On suppose que vous avez défini vos classes SVIParams, MarketDataPoint, SVICalibrationParams
# et éventuellement OptionDataParser dans votre module data_management.manage_bloomberg_data.
# Ici nous nous concentrons sur la construction de la surface de volatilité à partir de SPX data.

def convert_maturity(maturity_str):
    """
    Convertit une chaîne représentant la maturité en fraction d'année.
    Exemples:
      "1W" -> 1/52
      "5W" -> 5/52
      "1M" -> 1/12
      "10M" -> 10/12
      "1Y" -> 1.0
      "2Y" -> 2.0
    """
    maturity_str = str(maturity_str).strip().upper()
    if maturity_str.endswith('W'):
        try:
            weeks = float(maturity_str[:-1])
            return weeks / 52.0
        except:
            return np.nan
    elif maturity_str.endswith('M'):
        try:
            months = float(maturity_str[:-1])
            return months / 12.0
        except:
            return np.nan
    elif maturity_str.endswith('Y'):
        try:
            years = float(maturity_str[:-1])
            return years
        except:
            return np.nan
    else:
        try:
            return float(maturity_str)
        except:
            return np.nan


class OptionVolSurface:
    def __init__(self, data):
        """
        Initialise la surface de volatilité à partir d'un DataFrame.
        Le DataFrame doit contenir les colonnes : 'maturity', 'strike', 'vol'
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
                    vol_matrix[i, :] = np.nanmean(row)
        return vol_matrix

    def get_vol_smile(self, maturity):
        """
        Retourne le smile (volatilité en fonction du strike) pour une maturité donnée.
        Si la maturité n'est pas présente, une interpolation sur l'axe maturité est effectuée.
        :param maturity: maturité (en années)
        :return: (strikes, vols)
        """
        if maturity in self.maturities:
            index = np.where(self.maturities == maturity)[0][0]
            return self.strikes, self.vol_matrix[index, :]
        else:
            vols_interpolated = []
            for j, k in enumerate(self.strikes):
                f = interp1d(self.maturities, self.vol_matrix[:, j], bounds_error=False, fill_value="extrapolate")
                vols_interpolated.append(f(maturity))
            return self.strikes, np.array(vols_interpolated)

    def plot_vol_smile(self, maturity):
        strikes, vols = self.get_vol_smile(maturity)
        plt.figure(figsize=(8, 5))
        plt.plot(strikes, vols, marker='o', linestyle='-')
        plt.xlabel('Strike')
        plt.ylabel('Volatilité Implicite')
        plt.title(f'Volatility Smile pour maturité {maturity:.2f} an(s)')
        plt.grid(True)
        plt.show()

    def plot_vol_surface(self):
        X, Y = np.meshgrid(self.strikes, self.maturities)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.vol_matrix, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturité (années)')
        ax.set_zlabel('Volatilité Implicite')
        ax.set_title('Surface de Volatilité')
        plt.show()


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data_options")
    file_path = os.path.join(DATA_PATH, "clean_data_SPX.xlsx")
    # Lecture des données au format wide
    df_raw = pd.read_excel(file_path)
    # Transformation melt : la première colonne "Maturity" reste, les autres colonnes deviennent "strike" et "iv"
    df_melt = df_raw.melt(id_vars="Maturity", var_name="strike", value_name="iv")
    # Conversion de la maturité (ex: "1W", "1M", "2Y", etc.) en années
    df_melt["Maturity"] = df_melt["Maturity"].apply(convert_maturity)
    # Conversion des colonnes strike et iv en float
    df_melt["strike"] = df_melt["strike"].astype(float)
    df_melt["iv"] = df_melt["iv"].astype(float)
    # On renomme pour que le DataFrame possède les colonnes attendues par OptionVolSurface
    df_melt.rename(columns={"Maturity": "maturity", "iv": "vol"}, inplace=True)

    print("Données transformées :")
    print(df_melt.head())

    # Création de la surface de volatilité à partir des données SPX
    vol_surface_obj = OptionVolSurface(df_melt)

    # Choix d'une maturité pour afficher le smile : par exemple, la maturité médiane
    sorted_maturities = vol_surface_obj.maturities
    chosen_maturity = sorted_maturities[len(sorted_maturities) // 2]
    print(f"Maturité choisie pour le smile : {chosen_maturity:.2f} an(s)")

    # Affichage du smile de volatilité pour la maturité choisie
    vol_surface_obj.plot_vol_smile(chosen_maturity)

    # Affichage de la surface 3D de volatilité
    vol_surface_obj.plot_vol_surface()
