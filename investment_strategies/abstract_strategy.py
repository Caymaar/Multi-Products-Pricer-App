from abc import ABC, abstractmethod
from market.day_count_convention import DayCountConvention
from option.option import OptionPortfolio
from typing import Optional

# ---------------- Strategy Class Vanille ----------------
class Strategy(ABC):
    def __init__(self, name, pricing_date, maturity_date, convention_days="Actual/365"):
        """
        :param name: Nom de la stratégie
        :param pricing_date: Date de pricing
        :param maturity_date: Date de maturité
        :param convention_days: Convention de marché pour les jours de l'année
        """
        self.name = name
        self.pricing_date = pricing_date
        self.maturity_date = maturity_date
        self.dcc = DayCountConvention(convention_days)
        self.ttm = self.dcc.year_fraction(self.pricing_date, self.maturity_date)  # Time to maturity en années
        # Initialisation des attributs des options et poids
        self.options = []
        self.weights = []


    @abstractmethod
    def get_legs(self):
        """
        Retourne la liste des tuples (actif, poids) qui composent la stratégie.
        Exemple : [(option1, -1), (option2, 1)]
        """
        pass

    def _populate_legs(self):
        """
        Remplit les attributs options et weights avec les valeurs extraites de get_legs.
        Cette méthode est appelée lors de l'initialisation.
        """
        for option, weight in self.get_legs():
            self.options.append(option)
            self.weights.append(weight)

    def get_options_and_weights(self):
        """
        Retourne deux listes :
        - Une liste des options (ou actifs),
        - Une liste des poids associés à ces options.
        """
        options = []
        weights = []

        for option, weight in self.get_legs():
            options.append(option)
            weights.append(weight)

        return options, weights

    def price(self, engine, type: Optional[str] = "MC"):
        """
        Calcule le prix de la stratégie en agrégeant le prix de chaque actif pondéré par son poids.
        Cas sur stratégies vanilles, les autres stratégies étant surchargées.
        """
        options, weights = self.get_options_and_weights()
        ptf = OptionPortfolio(options, weights)
        engine.options = ptf
        return engine.price(type)