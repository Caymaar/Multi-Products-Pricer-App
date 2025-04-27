from pricers.mc_pricer import Engine, MonteCarloEngine
from typing import Union
from numpy import np
import copy
from scipy.stats import norm
from risk_metrics.greeks import GreeksCalculator

class Backtest:

    def __init__(self, strategy: Engine, pricing_date: str):

        self.loss = Loss(strategy, pricing_date)

    def run(self, var_type: str, alpha: Union[float, list[float]], **kwargs):

        var_func = getattr(self.loss, f"VaR_{var_type}")
        if isinstance(alpha, list):
            return [var_func(self.loss._original_model.pricing_date, alpha=a, **kwargs) for a in alpha]
        else:
            return var_func(self.loss._original_model.pricing_date, alpha=alpha, **kwargs)

class Loss:
    def __init__(self, strategy: Engine, pricing_date: float):

        self._original_model = copy.copy(strategy)
        self._retrieve_parameters(pricing_date)

    def _retrieve_parameters(self, pricing_date: str):
        self.S0 = self._original_model.market.S0
        self.r = self._original_model.market.r
        self.sigma = self._original_model.market.sigma
        self.T = self._original_model.T
        self.delta_T = self._recreate_model(pricing_date=pricing_date).T
        self.sigma_var = self.sigma * np.sqrt(self.delta_T)

    def _generate_St(self, alpha: float = 0.05, nb_simu: Union[None, int] = None) -> Union[float, np.ndarray]:

        St = self.S0 * np.exp(((self.r - 0.5 * self.sigma ** 2) * (self.T) + self.sigma * np.sqrt(self.delta_T) * norm.ppf(1 - alpha)))

        return St
    
    def VaR_TH(self, pricing_date: str, alpha: float = 0.05):
        
        St = self._generate_St(pricing_date, alpha)

        value_dt = self._recreate_model(pricing_date=pricing_date, market=self._original_model.market.copy(S0=St)).price()
        value_t = self._original_model.price()
        return value_dt - value_t

    def VaR_MC(self, pricing_date: str, alpha: float = 0.05, nb_simu: int = 100):
        
        St = self._generate_St(pricing_date, alpha, nb_simu)
        
        value_dt = np.vectorize(self._recreate_model)(pricing_date=pricing_date, market=self._original_model.market.copy(S0=St)).price()
        value_t = self._original_model.price()
        
        return np.percentile(value_dt * np.exp(-self.strategy.market.r * (self.strategy.T) - self._calculate_delta_T(self.delta_t)) - value_t,(1 - alpha) * 100)

    def VaR_CF(self, pricing_date: str, alpha: float = 0.05, order: int = 1):
        
        greeks = GreeksCalculator(self._original_model)

        delta = greeks.delta()
        gamma = greeks.gamma() if order >= 2 else 0
        speed = greeks.speed() if order >= 3 else 0

        first_order = self._get_first_order(delta, gamma, speed)
        second_order = self._get_second_order(delta, gamma, speed)
        third_order = self._get_third_order(delta, gamma, speed)
        fourth_order = self._get_fourth_order(delta, gamma, speed)

        variance = second_order **2
        kurtosis = ((fourth_order - 4 * third_order * first_order + 6 * second_order * first_order **2 - 3 * first_order **4)/np.sqrt(variance) **4) - 3
        skewness = (third_order - 3 * second_order * first_order **3) / np.sqrt(variance) **3

        z_a = norm.ppf(alpha)
        z_tilde = z_a + (z_a **2 - 1) * skewness / 6 

        return - (first_order + np.sqrt(variance) * z_tilde)


    def _get_first_order(self, delta: float, gamma: float, speed: float):
        return 1/2 * gamma * self.S0 **2 * self.sigma_var **2
    
    def _get_second_order(self, delta: float, gamma: float, speed: float):
        return  (1/36 * speed **2 * self.S0 **6) * 15 * self.sigma_var **6 + \
                ((1/4 * gamma **2 * self.S0 **4)+(1/3 * speed*self.S0 **3)*(delta * self.S0))* \
                3 * (self.sigma_var) **4 + \
                (delta **2 * self.S0 **2) * (self.sigma_var) **2
    
    def _get_third_order(self, delta: float, gamma: float, speed: float):
        return (3/2 * gamma * self.S**2) * (1/36 * speed**2 * self.S**6) * 105 * self.sigma_var**8 + \
                ((1/8 * gamma**3 * self.S**6) + (speed * self.S**3) * (1/2 * gamma * self.S**2) * \
                (delta * self.S)) * 15 * self.sigma_var**6 + \
                (3/2 * gamma * self.S**2) * \
                (delta**2 * self.S**2) * 3 * self.sigma_var**4

    def _get_fourth_order(self, delta: float, gamma: float, speed: float):
        return ((1/1296) * speed**4 * self.S**12) * 10395 * self.sigma_var**12 + \
                ((1/6 * speed**2 * self.S**6) * (1/4 * gamma**2 * self.S**4) + 4 * (delta * self.S) * \
                (1/216 * speed**3 * self.S**9)) * 945 * self.sigma_var**10 + \
                ((1/16 * gamma**4 * self.S**8) + (2 * speed * self.S**3) * (delta * self.S) * \
                (1/4 * gamma**2 * self.S**4) + (1/6 * speed**2 * self.S**6) * (delta**2 * self.S**2)) * \
                105 * self.sigma_var**8 + \
                ((3/2 * gamma**2 * self.S**4) * (delta**2 * self.S**2) + (2/3 * speed * self.S**3) * \
                (delta**3 * self.S**3)) * 15 * self.sigma_var**6 + \
                (delta**4 * self.S**4) * 3 * self.sigma_var**4