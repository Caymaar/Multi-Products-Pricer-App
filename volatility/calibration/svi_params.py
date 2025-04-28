from dataclasses import dataclass
from typing import List

# ---------------- SVIParams Class ----------------
@dataclass
class SVIParams:
    strike: float
    maturity: float
    spot: float

@dataclass
class MarketDataPoint:
    strike: float
    maturity: float
    implied_volatility: float

@dataclass
class SVICalibrationParams:
    opt_data: List[MarketDataPoint]
    spot: float
