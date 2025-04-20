import numpy as np
from scipy.stats import norm

from option.option import OptionPortfolio
from pricers.pricing_model import Engine
from pricers.regression import Regression
from stochastic_process.gbm_process import GBMProcess
from market.market import Market

# ---------------- Classe MCModel ----------------
class MonteCarloEngine(Engine):
    def __init__(self, market: Market, option_ptf: OptionPortfolio, pricing_date: datetime, n_paths: int, n_steps: int,
                 seed=None, ex_frontier="Quadratic",compute_antithetic=False):
        super().__init__(market, option_ptf, pricing_date, n_steps)
        self.n_paths = n_paths
        self.reg_type = ex_frontier
        self.eu_payoffs = None
        self.am_payoffs = None
        self.american_price_by_time = None

        self.GBMProcess = {}
        for option, dt in zip(self.options, self.dt):
            self.GBMProcess[option.name] = GBMProcess(market=self.market, dt=dt, n_paths=self.n_paths, n_steps=self.n_steps,
                                                     t_div=self.t_div, compute_antithetic=compute_antithetic, seed=seed)

    def _price_american_lsm(self, paths, analysis=False):
        global price_by_time

        # Payoff final brut
        CF = self.options.intrinsic_value(paths)

        # --- Analyse du pricing backward dans LSM ---
        if analysis:
            price_by_time = CF.mean() * np.exp(-self.market.r * self.T)

        # --- Backward induction ---
        for t in range(self.n_steps - 1, 0, -1):
            CF *= self.df  # actualisation

            # Valeur immédiate
            immediate = [option.intrinsic_value(paths[:, :t + 1]) for option in self.options]

            in_money = (immediate > 0)

            if np.any(in_money):
                X = paths[:, t][in_money]
                Y = CF[in_money]

                cont_val = Regression.fit(self.reg_type, X, Y)
                exercise = immediate[in_money] >= cont_val

                # Mise à jour des cashflows
                CF[in_money] = np.where(exercise, immediate[in_money], Y)

            if analysis:
                price_by_time.append(CF.mean() * np.exp(-self.market.r * (t) * self.dt))

        # Stocke le résultat final
        self.am_payoffs = CF

        if analysis:
            self.american_price_by_time = price_by_time

        return CF.mean()

    def _discounted_payoffs_by_method(self, type):
        """Calcule les payoffs associés à la méthode de pricing"""
        if type == "MC":
            if self.eu_payoffs is None:
                self.eu_payoffs = self._discounted_eu_payoffs(self.gbm_ptf_simulations())
            return self.eu_payoffs.copy()
        else:
            if self.am_payoffs is None:
                self._price_american_lsm(self.GBMProcess.simulate())
            return self.am_payoffs.copy()

    def _discounted_eu_payoffs(self, paths):
        """Calcule les payoffs actualisés pour un pricing européen."""
        payoffs = {}

        for id, option in enumerate(self.options):
            payoffs[option.name] = option.intrinsic_value(paths[option.name]) # Payoff à maturité
            payoffs[option.name] *= np.exp(-self.market.r * self.T[id]) # Actualisation

        return payoffs

    def get_variance(self, type="MC"):
        """Calcule la variance des payoffs actualisés pour la méthode de prix associée"""
        discounted_payoffs = self._discounted_payoffs_by_method(type)

        var_discounted_payoffs = {}

        for option in self.options:
            if self.GBMProcess[option.name].compute_antithetic:
                discounted_payoffs[option.name] = (discounted_payoffs[option.name][:self.n_paths//2] + discounted_payoffs[option.name][self.n_paths//2:]) / 2

            else:
                var_discounted_payoffs[option.name] = discounted_payoffs[option.name].var()

        return var_discounted_payoffs

    def get_american_price_path(self):
        #if self.american_price_by_time is None:
        self._price_american_lsm(self.GBMProcess.simulate(),analysis=True)
        return self.american_price_by_time


    def price_confidence_interval(self, alpha=0.05, type="MC"):
        """Calcule le prix et son intervalle de confiance Monte Carlo."""
        discounted_payoffs = self._discounted_payoffs_by_method(type)
           
        # Récupère les payoffs actualisés
        mean_price = [np.mean(discounted_payoffs[payoff]) for payoff in discounted_payoffs] # Prix moyens estimés
        var_price = self.get_variance() # Variances des payoffs
        std_dev = [np.sqrt(var_price[option]) for option in var_price]  # Écart-types des payoffs

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

    def european_price(self) -> np.ndarray:
        #simulations = [self.GBMProcess[option.name].simulate() for option in self.options]
        paths = self.gbm_ptf_simulations()
        payoffs = self._discounted_eu_payoffs(paths)
        prices = np.array([np.mean(payoffs[ind_payoff]) for ind_payoff in payoffs])

        return prices

    def american_price(self):
        return self._price_american_lsm(self.GBMProcess.simulate())

    def gbm_ptf_simulations(self) -> dict:
        """
        Simule les trajectoires de l'ensemble du portefeuille d'options.
        :return: Un dictionnaire contenant les trajectoires simulées pour chaque option.
        """
        # Simule les trajectoires pour chaque option
        simulations = {}
        for option in self.options:
            simulations[option.name] = np.array([self.GBMProcess[option.name].simulate()])

        return simulations

    def _recreate_model(self, **kwargs):
        """
        Recrée une instance du modèle avec des paramètres mis à jour.
        :param kwargs: Les paramètres à mettre à jour dans le nouveau modèle.
        :return: Une nouvelle instance de MonteCarloEngine.
        """
        new_params = {
            "market": self.market.copy(),  # Copie du marché actuel
            "option_ptf": self.option_ptf,
            "pricing_date": self.pricing_date,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "seed": self.GBMProcess.brownian.seed if hasattr(self.GBMProcess.brownian, 'seed') else None  # Copie la graine si disponible
            }
        new_params.update(kwargs)  # Mise à jour avec les paramètres fournis
        return MonteCarloEngine(**new_params)