import numpy as np
from typing import Any, List, Tuple, Union

# Pas fini mais c'est l'idée d'avoir un portefeuille d'options et de calculer la sensibilité

# ---------------- SensitivityAnalyzer Class ----------------
class SensitivityAnalyzer:
    def __init__(self, portfolio: Any, engine: Any) -> None:
        """
        Initialise l'analyseur de sensibilité.

        :param portfolio: Objet stratégie possédant une méthode get_legs() et price(engine).
        :param engine: Moteur de pricing utilisé pour évaluer les options.
        """
        self.portfolio = portfolio
        self.engine = engine

    def underlying_sensitivity(
        self,
        underlying_range: Union[np.ndarray, List[float]]
    ) -> Tuple[Union[np.ndarray, List[float]], np.ndarray]:
        """
        Calcule l'évolution de la valeur du portefeuille lorsque le sous-jacent varie.

        :param underlying_range: Tableau numpy ou liste des valeurs du sous-jacent à tester.
        :return: Tuple (underlying_range, portfolio_values) où portfolio_values est un tableau numpy.
        """
        portfolio_values: List[float] = []
        # Pour chaque valeur du sous-jacent, mettre à jour les options du portefeuille
        for S in underlying_range:
            for option, weight in self.portfolio.get_legs():
                # On suppose que chaque option possède un attribut S pour le prix du sous-jacent
                option.S = S
            portfolio_values.append(self.portfolio.price(self.engine))
        return underlying_range, np.array(portfolio_values)

    def volatility_sensitivity(
        self,
        volatility_range: Union[np.ndarray, List[float]]
    ) -> Tuple[Union[np.ndarray, List[float]], np.ndarray]:
        """
        Calcule l'évolution de la valeur du portefeuille lorsque la volatilité varie.

        :param volatility_range: Tableau numpy ou liste des valeurs de volatilité à tester.
        :return: Tuple (volatility_range, portfolio_values) où portfolio_values est un tableau numpy.
        """
        portfolio_values: List[float] = []
        for vol in volatility_range:
            for option, weight in self.portfolio.get_legs():
                # On suppose que chaque option possède un attribut volatility
                option.volatility = vol
            portfolio_values.append(self.portfolio.price(self.engine))
        return volatility_range, np.array(portfolio_values)

    def maturity_sensitivity(
        self,
        maturity_range: Union[np.ndarray, List[float]]
    ) -> Tuple[Union[np.ndarray, List[float]], np.ndarray]:
        """
        Calcule l'évolution de la valeur du portefeuille lorsque le temps à maturité varie.

        :param maturity_range: Tableau numpy ou liste des valeurs de maturité à tester.
        :return: Tuple (maturity_range, portfolio_values) où portfolio_values est un tableau numpy.
        """
        portfolio_values: List[float] = []
        for T in maturity_range:
            for option, weight in self.portfolio.get_legs():
                # On met à jour le temps à maturité de chaque option
                option.T = T
            portfolio_values.append(self.portfolio.price(self.engine))
        return maturity_range, np.array(portfolio_values)
