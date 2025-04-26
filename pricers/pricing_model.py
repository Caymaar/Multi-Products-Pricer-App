from abc import ABC
from market.market import Market
from option.option import OptionPortfolio
from pricers.bs_pricer import BlackScholesPricer
import numpy as np
import datetime
from typing import Dict

# ---------------- Model Abstract Class ----------------
class Engine(ABC):
    def __init__(self, market: Market, option_ptf: OptionPortfolio, pricing_date: datetime, n_steps: int):
        self.market = market
        self._options = option_ptf.options
        self.pricing_date = pricing_date
        self.n_steps = n_steps
        self.T = self._calculate_T()
        self.dt = self.T / n_steps
        self.df = np.exp(-self.market.r * self.dt)
        self.t_div = self._calculate_t_div()
        # ** instanciation des BS pricer **
        self._init_bsm_pricers()

    def _calculate_T(self):
        """
        Méthode pour calculer les temps jusqu'à l'expiration des options (T).
        Retourne un array des durées en années entre la date de pricing et les dates de maturité.
        """
        return np.array([self.market.DaysCountConvention.year_fraction(start_date=self.pricing_date, end_date=option.T) for option in self.options])

    def _calculate_t_div(self):
        """
        Calcule l'indice temporel pour le dividende si le dividende est discret.
        """
        if self.market.div_type == "discrete" and self.market.div_date is not None:
            T_div = self.market.DaysCountConvention.year_fraction(start_date=self.pricing_date,end_date=self.market.div_date) # Conversion en année
            return int(T_div/self.dt)  # Conversion en indice de pas de temps
        else:
            return None

    def _init_bsm_pricers(self):
        """
        Pour chaque option du portefeuille, crée un
        BlackScholesPricer avec les bons T_i et dt_i.
        """
        self.bsm_pricers: Dict[str, BlackScholesPricer] = {}
        for opt, T_i, dt_i in zip(self._options, self.T, self.dt):
            # t_div reste le même pour toutes les options (indice de dividende)
            pricer = BlackScholesPricer(
                market=self.market,
                option=opt,
                t_div=self.t_div,
                dt=dt_i,
                T=T_i
            )
            self.bsm_pricers[opt.name] = pricer

    def price_bs(self) -> np.ndarray:
        """
        Retourne un array des prix Black-Scholes pour chaque option du portefeuille.
        """
        prices = []
        for opt in self._options:
            pr = self.bsm_pricers[opt.name].price()
            prices.append(pr)
        return np.array(prices)

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, new_options):
        self._options = new_options
        self.T = self._calculate_T()
        self.dt = self.T / self.n_steps
        self.t_div = self._calculate_t_div()
        self._init_bsm_pricers()