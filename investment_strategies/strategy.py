from abc import ABC, abstractmethod
from market.day_count_convention import DayCountConvention

# ---------------- Strategy Class Vanille ----------------
class Strategy(ABC):
    def __init__(self, name, pricing_date, maturity_date, convention_days="Actual/365"):
        """
        :param name: Nom de la stratégie
        """
        self.name = name
        self.pricing_date = pricing_date
        self.maturity_date = maturity_date
        self.dcc = DayCountConvention(convention_days)
        self.ttm = self.dcc.year_fraction(self.pricing_date, self.maturity_date)  # Time to maturity en années

    @abstractmethod
    def get_legs(self):
        """
        Retourne la liste des tuples (actif, poids) qui composent la stratégie.
        Exemple : [(option1, -1), (option2, 1)]
        """
        pass

    def price(self, engine):
        """
        Calcule le prix de la stratégie en agrégeant le prix de chaque actif pondéré par son poids.
        """
        total_price = 0
        for option, weight in self.get_legs():
            option_price = engine.price(option)
            total_price += weight * option_price
        return total_price