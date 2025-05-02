from abc import ABC, abstractmethod
from typing import Any

# ---------------- Abstract Volatility Model ----------------
class VolatilityModel(ABC):
    @abstractmethod
    def get_volatility(self, parameters: Any) -> float:
        """
        Retourne la volatilité calculée par le modèle pour des paramètres donnés.
        """
        pass

    @abstractmethod
    def calibrate(self, calibration_params: Any) -> None:
        """
        Calibre le modèle sur des données de marché.
        """
        pass
