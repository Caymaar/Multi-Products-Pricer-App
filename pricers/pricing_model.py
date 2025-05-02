from abc import ABC
from market.market import Market
try:
    from option.option import OptionPortfolio
except:
    from option import OptionPortfolio
from pricers.bs_pricer import BSPortfolio
import numpy as np
import datetime


# ---------------- Model Abstract Class ----------------
class Engine(ABC):
    def __init__(self, market: Market, option_ptf: OptionPortfolio, pricing_date: datetime, n_steps: int):
        self.market = market
        self._options = option_ptf
        self.pricing_date = pricing_date
        self.n_steps = n_steps
        self.T = self._calculate_T()
        self.dt = self.T / n_steps
        self.t_div = self._calculate_t_div()
        # ** instanciation des BS pricer **
        self._init_bsm()

    def _calculate_T(self):
        """
        Méthode pour calculer les temps jusqu'à l'expiration des options (T).
        Retourne un array des durées en années entre la date de pricing et les dates de maturité.
        """
        return np.array([self.market.dcc.year_fraction(start_date=self.pricing_date, end_date=option.T) for option in self.options.assets])

    def _calculate_t_div(self):
        """
        Calcule l'indice temporel pour le dividende si le dividende est discret.
        """
        if self.market.div_type == "discrete" and self.market.div_date is not None:
            T_div = self.market.dcc.year_fraction(start_date=self.pricing_date,end_date=self.market.div_date) # Conversion en année
            return np.floor(T_div / self.dt).astype(int) # Conversion en indice de pas de temps
        else:
            return None

    def _init_bsm(self):
        """
        Crée un BSPortfolio
        """
        pricer = BSPortfolio(
            market=self.market,
            option_ptf=self._options,
            pricing_date=self.pricing_date,
        )
        self.bsm = pricer

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new_options : OptionPortfolio):
        self._options = new_options
        self.T = self._calculate_T()
        self.dt = self.T / self.n_steps
        self.t_div = self._calculate_t_div()
        self._init_bsm()