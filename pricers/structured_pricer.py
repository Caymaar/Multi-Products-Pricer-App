# a implémenter pour les autocall nécéssitant Monte Carlo

from mc_pricer import MonteCarloEngine
from market.market import Market
from option.option import Option
import numpy as np

class StructuredPricer(MonteCarloEngine):

    def __init__(self, market: Market, option: Option, pricing_date, n_paths, n_steps):
        super().__init__(market, option, pricing_date, n_paths, n_steps)
        self.option = option
        self.pricing_date = pricing_date
        self.n_paths = n_paths
        self.n_steps = n_steps

    @property
    def option_price(self):
        return self.price(type='MC')

    @property
    def zc_price(self):
        return self.price(type='ZC')