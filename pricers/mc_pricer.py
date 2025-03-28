import numpy as np
from scipy.stats import norm
from pricers.pricing_model import Engine
from pricers.regression import Regression
from stochastic_process.gbm_process import GBMProcess

# ---------------- Classe MCModel ----------------
class MonteCarloEngine(Engine):
    def __init__(self, market, option, pricing_date, n_paths, n_steps, seed=None, ex_frontier="Quadratic",compute_antithetic=False):
        super().__init__(market, option, pricing_date, n_steps)
        self.n_paths = n_paths
        self.reg_type = ex_frontier
        self.eu_payoffs = None
        self.am_payoffs = None
        self.american_price_by_time = None
        self.GBMProcess = GBMProcess(
            self.market, self.dt, self.n_paths, self.n_steps, self.t_div, compute_antithetic, seed)

    def _price_american_lsm(self, paths, analysis=False):
        global price_by_time

        CF = self.option.payoff(paths[:,-1]) # Valeur du payoff à l'échéance

        if analysis:
            price_by_time = []
            price_by_time.append(CF.mean() * np.exp(-self.market.r * self.T))

        for t in range(self.n_steps - 2, -1, -1):

            CF *= self.df # Actualisation en un seul calcul
            immediate = self.option.payoff(paths[:, t])
            in_money = immediate > 0  # Mask des options ITM

            if np.any(in_money):  # Vérifie si au moins une option est ITM
                cont_val = Regression.fit(self.reg_type, paths[in_money, t], CF[in_money])
                exercise = immediate[in_money] >= cont_val
                CF[in_money] = np.where(exercise, immediate[in_money], CF[in_money])

                if analysis:
                    price_by_time.append(CF.mean() * np.exp(-self.market.r * (t + 1) * self.dt))

        self.am_payoffs = CF

        if analysis:
            self.american_price_by_time = price_by_time

        return CF.mean()

    def _discounted_payoffs_by_method(self, type):
        """Calcule les payoffs associés à la méthode de pricing"""
        if type == "MC":
            if self.eu_payoffs is None:
                self.eu_payoffs = self._discounted_eu_payoffs(
                    self.GBMProcess.simulate())
            return self.eu_payoffs.copy()
        else:
            if self.am_payoffs is None:
                self._price_american_lsm(self.GBMProcess.simulate())
            return self.am_payoffs.copy()

    def _discounted_eu_payoffs(self, paths):
        """Calcule les payoffs actualisés pour un pricing européen."""
        payoffs = self.option.payoff(paths[:, -1])  # Payoff à maturité
        return np.exp(-self.market.r * self.T) * payoffs # Actualisation
      
    def get_variance(self, type="MC"):
        """Calcule la variance des payoffs actualisés pour la méthode de prix associée"""
        discounted_payoffs = self._discounted_payoffs_by_method(type)

        if self.GBMProcess.compute_antithetic:
            discounted_payoffs = (discounted_payoffs[:self.n_paths//2] + discounted_payoffs[self.n_paths//2:]) / 2
        return discounted_payoffs.var()

    def get_american_price_path(self):
        #if self.american_price_by_time is None:
        self._price_american_lsm(self.GBMProcess.simulate(),analysis=True)
        return self.american_price_by_time
    
    def _european_price(self, paths):
        """Calcule le prix européen moyen."""
        payoffs=self._discounted_eu_payoffs(paths)
        return np.mean(payoffs)

    def price_confidence_interval(self, alpha=0.05, type="MC"):
        """Calcule le prix et son intervalle de confiance Monte Carlo."""
        discounted_payoffs = self._discounted_payoffs_by_method(type)
           
         # Récupère les payoffs actualisés
        mean_price = np.mean(discounted_payoffs)  # Prix moyen estimé

        std_dev = np.sqrt(self.get_variance().copy())  # Écart-type des payoffs

        # Quantile de la loi normale pour l'intervalle de confiance (avec numpy)
        z = norm.ppf(1 - alpha / 2)  # Approximation sans scipy

        # Calcul de la marge d'erreur
        CI_half_width = z * (std_dev / np.sqrt(self.n_paths))

        CI_lower = mean_price - CI_half_width
        CI_upper = mean_price + CI_half_width

        return (CI_upper, CI_lower)
    
    def price(self, type="MC"):
        """ Retourne le prix associé au type d'option enregistré"""
        if type == "Longstaff":
            return self.american_price()
        else:
            return self.european_price()

    def european_price(self):
        return self._european_price(self.GBMProcess.simulate())

    def american_price(self):
        return self._price_american_lsm(self.GBMProcess.simulate())