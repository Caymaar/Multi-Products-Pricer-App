from typing import Any
from ..abstract_volatility import VolatilityModel
from ..calibration.cste_calibration_params import CsteCalibrationParams

class CsteVolatility(VolatilityModel):
    def __init__(self) -> None:
        # Initialisation par défaut de la volatilité constante à 0.0
        self.vol_level: float = 0.0

    def calibrate(self, calibration_params: Any) -> None:
        """
        Calibre le modèle en assignant directement le niveau de volatilité passé.
        :param calibration_params: Instance de CsteCalibrationParams contenant le niveau de volatilité.
        """
        if not isinstance(calibration_params, CsteCalibrationParams):
            raise TypeError("Expected calibration_params of type CsteCalibrationParams")
        self.vol_level = calibration_params.vol_level

    def get_volatility(self, parameters: Any = None) -> float:
        """
        Retourne la volatilité constante calibrée.
        :param parameters: Ignoré pour ce modèle.
        :return: Niveau de volatilité.
        """
        return self.vol_level