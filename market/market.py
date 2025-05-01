from datetime import datetime
from market.day_count_convention import DayCountConvention
from typing import Callable, Optional
import pandas as pd
import numpy as np

class Market:
    def __init__(self,
                 S0: float,
                 sigma: float,
                 div_type: str = "continuous",
                 dividend: float = 0,
                 div_date: datetime | None = None,
                 day_count: str = "Actual/365",
                 discount_curve: Callable[[float], float] | float | None = None,
                 forward_curve:  Callable[[float, float], float] | None = None,
                 zero_rate_curve :Callable[[float], float] | float | None = None,
                 corr_matrix:    Optional[Callable[[datetime, datetime], pd.DataFrame]] = None
                ):
        self.S0        = S0
        self.sigma     = sigma
        self.dividend  = dividend
        self.div_type  = div_type
        self.div_date  = div_date
        self.dcc       = DayCountConvention(day_count)
        self.corr_matrix = corr_matrix

        # — ZERO RATE CURVE —
        # si on reçoit un float, on le wrappe dans une function constante
        if isinstance(zero_rate_curve, (int, float)):
            zr = float(zero_rate_curve)
            self.zero_rate = lambda t: zr
        elif zero_rate_curve is None:
            # par défaut taux zéro nul
            self.zero_rate = lambda t: 0.0
        else:
            self.zero_rate = zero_rate_curve

        # — DISCOUNT CURVE —
        # si on reçoit un float, on wrappe ; sinon si None, on la dérive de zero_rate_curve
        if isinstance(discount_curve, (int, float)):
            dfc = float(discount_curve)
            self.discount = lambda t: dfc
        elif discount_curve is None:
            self.discount = lambda t: np.exp(-self.zero_rate(t) * t)
        else:
            self.discount = discount_curve

        # — FORWARD CURVE —
        # si on reçoit un float, wrap ; sinon si None, on calcule le forward implicite discret
        if isinstance(forward_curve, (int, float)):
            fc = float(forward_curve)
            self.forward = lambda t1, t2: fc
        elif forward_curve is None:
            def _fwd(t1: float, t2: float) -> float:
                d1 = self.discount(t1)
                d2 = self.discount(t2)
                return (d1 / d2 - 1) / (t2 - t1)

            self.forward = _fwd
        else:
            self.forward = forward_curve

    def discount_factor(self, t: float) -> float:
        return self.discount(t)

    def forward_rate(self, t1: float, t2: float) -> float:
        return self.forward(t1, t2)

    def zero_rate(self, t: float) -> float:
        return self.zero_rate(t)

    def price_rate_correlation(self, pricing_date:datetime, maturity: datetime) -> Optional[pd.DataFrame]:
        """
        Renvoie la matrice de corrélation pour la maturité donnée,
        en appelant correlation_fn(pricing_date, maturity).
        """
        if self.corr_matrix is None:
            return None
        return self.corr_matrix(pricing_date, maturity)

    def copy(self, **overrides) -> "Market":
        """
        Retourne une nouvelle instance de Market en reprenant
        tous les attributs actuels, sauf ceux passés en overrides.
        Exemples d'overrides : S0=..., sigma=..., day_count="30/360", discount_curve=...
        """
        params = {
            "S0":               self.S0,
            "sigma":            self.sigma,
            "dividend":         self.dividend,
            "div_type":         self.div_type,
            "div_date":         self.div_date,
            "day_count":        self.dcc.convention,     # récupère la convention initiale
            "discount_curve":   self.discount,
            "forward_curve":    self.forward,
            "zero_rate_curve":  self.zero_rate,
            "corr_matrix":      self.corr_matrix
        }
        params.update(overrides)
        return Market(**params)
