from market.day_count_convention import DayCountConvention

# ---------------- Market Class ----------------
class Market:
    def __init__(self, S0, r, sigma, dividend, div_type="continuous", div_date=None, days_convention="Actual/365"):
        self.S0 = S0  # Prix initial du sous-jacent
        self.r = r  # Taux d'intérêt sans risque
        self.sigma = sigma  # Volatilité
        self.dividend = dividend  # Dividende (yield / discret)
        self.div_date = div_date
        self.div_type = div_type.lower()
        self.DaysCountConvention = DayCountConvention(days_convention)
        if self.div_type not in ["continuous", "discrete"]:
            raise ValueError("div_type must be 'continuous' or 'discrete'")

    def copy(self, **kwargs):
        """Crée une copie du marché avec des valeurs modifiées."""
        params = self.__dict__.copy()  # Copie les attributs existants
        params.pop("DaysCountConvention", None)  # Exclut DaysCountConvention
        params.update(kwargs)  # Applique les modifications
        return Market(**params)