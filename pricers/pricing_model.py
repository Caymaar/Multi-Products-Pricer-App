from abc import ABC
from market.market import Market
from option.option import Option, OptionPortfolio
from pricers.bs_pricer import BlackScholesPricer
import numpy as np

# ---------------- Model Abstract Class ----------------
class Engine(ABC):
    def __init__(self, market: Market, options: OptionPortfolio, pricing_date, n_steps: list):
        self.market = market
        # self._option = option
        self.options = options
        self.pricing_date = pricing_date
        self.n_steps = n_steps
        self.T = self._calculate_T()
        self.dt = np.array(self.T) / np.array(n_steps)
        self.df = np.exp(-self.market.r * self.dt)
        self.t_div = self._calculate_t_div()
        # Conversion de div_date en indice temporel si dividende discret
        self.bsm = BlackScholesPricer(self.market, self.option, self.t_div, self.dt, self.T)

    def _calculate_T(self):
        """
        Méthode pour calculer les temps jusqu'à l'expiration des options (T).
        Retourne une liste des durées en années entre la date de pricing et les dates de maturité.
        """
        return [self.market.DaysCountConvention.year_fraction(start_date=self.pricing_date, end_date=option.T) for option in self.options]

    def _calculate_t_div(self):
        """
        Calcule l'indice temporel pour le dividende si le dividende est discret.
        """
        if self.market.div_type == "discrete" and self.market.div_date is not None:
            T_div = self.market.DaysCountConvention.year_fraction(start_date=self.pricing_date,end_date=self.market.div_date) # Conversion en année
            return int(T_div/self.dt)  # Conversion en indice de pas de temps
        else:
            return None

    '''@property
    def option(self):
        return self._option

    @option.setter
    def option(self, new_option):
        T = self.option.T
        self._option = new_option
        self.bsm.option = new_option
        if new_option.T != T: # évites le recalcul des périodes si inchangé
            self.T = self.market.DaysCountConvention.year_fraction(start_date=self.pricing_date, end_date=self._option.T)
            self.dt = self.T / self.n_steps
            self.t_div = self._calculate_t_div()
            self.bsm = BlackScholesPricer(self.market, self.option, self.t_div, self.dt, self.T)'''