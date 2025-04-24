import numpy as np
from scipy.stats import norm
from datetime import datetime
from option.option import OptionPortfolio
from pricers.pricing_model import Engine
from pricers.regression import Regression
from stochastic_process.gbm_process import GBMProcess
from market.market import Market
from typing import List, Tuple, Dict


# ---------------- Classe MCModel ----------------
class MonteCarloEngine(Engine):
    def __init__(self, market: Market, option_ptf: OptionPortfolio, pricing_date: datetime, n_paths: int, n_steps: int,
                 seed=None, ex_frontier="Quadratic",compute_antithetic=True):
        super().__init__(market, option_ptf, pricing_date, n_steps)
        self.n_paths = n_paths
        self.reg_type = ex_frontier
        self.eu_payoffs = None
        self.am_payoffs = None
        self.american_price_by_time = None

        self.diffusions = {}  # Stocke les processus de diffusion pour chaque option
        for option, dt in zip(self.options, self.dt):
            self.diffusions[option.name] = GBMProcess(market=self.market, dt=dt, n_paths=self.n_paths, n_steps=self.n_steps,
                                                     t_div=self.t_div, compute_antithetic=compute_antithetic, seed=seed)

    def _price_american_lsm(self, paths, option, idx: int, analysis=False) -> Tuple[np.ndarray, List[float]]:
        """
        LSM backward pour *UNE* option à l'indice idx.
        :param paths: (n_paths, n_steps+1)
        :param option: instance d'Option
        :param idx: position de l'option dans self.options
        :param analysis: si True, collecte price_by_time
        :return: (CF0, price_by_time) où
                 CF0 est ndarray(n_paths,) actualisé à t=0,
                 price_by_time est list des prix moyens à chaque date.
        """

        global price_by_time

        # Extraire T_i, dt_i, df_i pour cette option
        T_i = self.T[idx]
        dt_i = T_i / self.n_steps
        df_i = np.exp(-self.market.r * dt_i)

        # Payoff final brut
        CF = option.intrinsic_value(paths)

        # --- Analyse du pricing backward dans LSM ---
        if analysis:
            price_by_time = CF.mean() * np.exp(-self.market.r * T_i)

        # --- Backward induction ---
        for t in range(self.n_steps - 1, 0, -1):
            CF *= df_i  # actualisation

            # Valeur immédiate
            immediate = option.intrinsic_value(paths[:, :t + 1])

            in_money = (immediate > 0)

            if np.any(in_money):
                X = paths[:, t][in_money]
                Y = CF[in_money]

                cont_val = Regression.fit(self.reg_type, X, Y)
                exercise = immediate[in_money] >= cont_val

                # Mise à jour des cashflows
                CF[in_money] = np.where(exercise, immediate[in_money], Y)

            if analysis:
                time_remaining = t * dt_i
                price_by_time.append(CF.mean() * np.exp(-self.market.r * time_remaining))

        CF = CF * df_i
        return CF, price_by_time

    def american_price(self) -> np.ndarray:
        """
        Calcule en LSM le prix américain de chaque option du portefeuille.
        Retourne un array de shape (n_options,).
        """
        # 1) Simulations pour tout le portefeuille, une seule fois
        paths_dict = self.gbm_ptf_simulations()
        #    -> { option.name : array (n_paths, n_steps+1) }
        self.am_payoffs = {}

        prices = np.zeros(len(self.options))
        # 2) Backward LSM option par option
        for idx, option in enumerate(self.options):
            paths = paths_dict[option.name]
            cf,_ = self._price_american_lsm(paths, option, idx)  # vecteur (n_paths,)
            self.am_payoffs[option.name] = cf.copy()  # on stocke pour usage ultérieur
            prices[idx] = cf.mean()

        return prices

    def _discounted_payoffs_by_method(self, type):
        """Calcule les payoffs associés à la méthode de pricing"""
        if type == "MC":
            if self.eu_payoffs is None:
                self.eu_payoffs = self._discounted_eu_payoffs(self.gbm_ptf_simulations())
            return self.eu_payoffs.copy()
        else:
            if self.am_payoffs is None:
                self.american_price()
            return self.am_payoffs.copy()

    def _discounted_eu_payoffs(self, paths):
        """Calcule les payoffs actualisés pour un pricing européen."""
        payoffs = {}

        for id, option in enumerate(self.options):
            payoffs[option.name] = option.intrinsic_value(paths[option.name]).astype(float) # Payoff à maturité
            payoffs[option.name] *= np.exp(-self.market.r * self.T[id]) # Actualisation

        return payoffs

    def get_variance(self, type="MC"):
        discounted = self._discounted_payoffs_by_method(type)
        var_out = {}
        for name, arr in discounted.items():
            if self.diffusions[name].compute_antithetic:
                half = self.n_paths // 2
                # antithétique
                arr = (arr[:half] + arr[half:]) / 2
            var_out[name] = arr.var(ddof=1)
        return var_out

    def get_american_price_path(self) -> Dict[str, List[float]]:
        """
        Retourne, pour chaque option du portefeuille, la trajectoire des prix moyens
        { option.name : [P(T), P(t_{n-1}), …, P(dt)] }.
        """
        paths_dict = self.gbm_ptf_simulations()
        price_paths: Dict[str, List[float]] = {}

        for idx, option in enumerate(self.options):
            paths = paths_dict[option.name]
            _, pbyt = self._price_american_lsm(paths, option, idx, analysis=True)
            price_paths[option.name] = pbyt

        return price_paths


    def price_confidence_interval(self, alpha=0.05, type="MC"):
        """Calcule le prix et son intervalle de confiance Monte Carlo."""
        discounted_payoffs = self._discounted_payoffs_by_method(type)
           
        # Récupère les payoffs actualisés
        mean_price = np.array([np.mean(discounted_payoffs[payoff]) for payoff in discounted_payoffs])  # Prix moyens estimés
        var_price = self.get_variance(type)  # Variances des payoffs
        std_dev = np.array([np.sqrt(var_price[option]) for option in var_price])  # Écart-types des payoffs

        # Quantile de la loi normale pour l'intervalle de confiance (avec numpy)
        z = norm.ppf(1 - alpha / 2)  # Approximation sans scipy

        # Calcul de la marge d'erreur
        CI_half_width = z * (std_dev / np.sqrt(self.n_paths))

        CI_lower = mean_price - CI_half_width
        CI_upper = mean_price + CI_half_width

        return CI_upper, CI_lower
    
    def price(self, type="MC"):
        """ Retourne le prix associé au type d'option enregistré"""
        if type == "Longstaff":
            return self.american_price()
        else:
            return self.european_price()

    def european_price(self) -> np.ndarray:
        paths = self.gbm_ptf_simulations()
        payoffs = self._discounted_eu_payoffs(paths)
        prices = np.array([np.mean(payoffs[ind_payoff]) for ind_payoff in payoffs])

        return prices


    def gbm_ptf_simulations(self) -> dict:
        """
        Simule les trajectoires de l'ensemble du portefeuille d'options.
        :return: Un dictionnaire contenant les trajectoires simulées pour chaque option.
        """
        # Simule les trajectoires pour chaque option
        simulations = {}
        for option in self.options:
            simulations[option.name] = self.diffusions[option.name].simulate()

        return simulations

    def _recreate_model(self, **kwargs) -> "MonteCarloEngine":
        """
        Recrée une nouvelle instance du MonteCarloEngine en reprenant
        tous les paramètres actuels, sauf ceux passés en kwargs.
        """
        # On prend un GBMProcess au hasard pour extraire seed, compute_antithetic et t_div
        proc = next(iter(self.diffusions.values()))

        base_params = {
            "market":          self.market.copy(),
            "option_ptf":      self.options,
            "pricing_date":    self.pricing_date,
            "n_paths":         self.n_paths,
            "n_steps":         self.n_steps,
            "seed":            proc.brownian.seed,
            "ex_frontier":     self.reg_type,
            "compute_antithetic": proc.compute_antithetic,
            "t_div":           proc.t_div,
        }

        # Surcharge avec ce que l'utilisateur fournit
        base_params.update(kwargs)

        # Création de la nouvelle instance
        return MonteCarloEngine(**base_params)
