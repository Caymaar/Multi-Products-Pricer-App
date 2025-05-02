from typing import Union
import numpy as np
import copy
from scipy.stats import norm
from pricers.bs_pricer import BlackScholesPricer, BSPortfolio
from risk_metrics.greeks import GreeksCalculator
from datetime import datetime, timedelta
from option.option import Call

class Backtest:
    def __init__(self, model, shift_date: datetime, p_type="MC"):

        self.loss = Loss(model, shift_date, p_type)

    def run(self, var_type: str, alpha: Union[float, list[float]], **kwargs):

        var_func = getattr(self.loss, f"VaR_{var_type}")
        if isinstance(alpha, list):
            return [var_func(alpha=a, **kwargs) for a in alpha]
        else:
            return var_func(alpha=alpha, **kwargs)

class Loss:
    def __init__(self, model, shift_date: datetime, p_type="MC"):

        self._original_model = copy.copy(model)
        self._retrieve_parameters(shift_date)
        self.shift_date = shift_date
        self.opt_weights = np.array(self._original_model.options.weights)
        self.p_type = p_type

    def _retrieve_parameters(self, shift_date: datetime):
        self.S0 = self._original_model.market.S0
        self.sigma = self._original_model.market.sigma
        self.pricing_date = self._original_model.pricing_date
        self.adj_T = self._original_model.market.dcc.year_fraction(start_date=self.pricing_date, end_date=shift_date)
        self.sigma_var = self.sigma * np.sqrt(self.adj_T)

    def _generate_St(self, alpha: float = 0.05, nb_simu: Union[None, int] = None) -> Union[float, np.ndarray]:

        if nb_simu is None:
            multiplier = norm.ppf(1 - alpha)
        else:
            multiplier = np.random.normal(0, 1, nb_simu)
        St = self.S0 * np.exp(
            ((self._original_model.market.zero_rate(self.adj_T) - 0.5 * self.sigma ** 2) * self.adj_T + self.sigma * np.sqrt(self.adj_T) * multiplier))

        return St
    
    def VaR_TH(self, alpha: float = 0.05):
        
        St = self._generate_St(alpha)

        value_dt = self._original_model.recreate_model(pricing_date=self.shift_date, market=self._original_model.market.copy(S0=St)).price(type=self.p_type)
        value_t = self._original_model.price(type=self.p_type)
        loss = (value_dt - value_t) * self.opt_weights
        return loss.sum() if loss.ndim > 0 else loss

    def VaR_MC(self, alpha: float = 0.05, nb_simu: int = 100):
        
        St = self._generate_St(alpha, nb_simu)

        value_dt = np.array([self._original_model.recreate_model(pricing_date=self.shift_date,market=self._original_model.market.copy(S0=s)).price(type=self.p_type)
                            for s in St])
        value_t = np.atleast_1d(self._original_model.price(type=self.p_type))  # toujours array 1D

        # Actualise la valeur future
        discount_factor = np.exp(-self._original_model.market.zero_rate(self.adj_T) * self.adj_T)
        value_dt = value_dt * discount_factor

        # Assure que value_dt est 2D (n_simu, n_options)
        if value_dt.ndim == 1:
            value_dt = value_dt[:, np.newaxis]

        percentiles = []
        for i, option in enumerate(self._original_model.options.assets):
            if isinstance(option, Call):
                p = np.percentile(value_dt[:, i], (1 - alpha) * 100)
            else:  # Put
                p = np.percentile(value_dt[:, i], alpha * 100)
            percentiles.append(p)

        percentiles = np.array(percentiles)

        # Maintenant calcul final
        results = (percentiles - value_t) * self.opt_weights

        return results.sum() if results.ndim > 0 else results

    def VaR_CF(self, alpha: float = 0.05, order: int = 1):

        if isinstance(self._original_model, Union[BSPortfolio, BlackScholesPricer]):
            greeks = self._original_model
        else:
            greeks = GreeksCalculator(self._original_model)

        delta = -np.sum(greeks.delta() * self.opt_weights)
        gamma = -np.sum(greeks.gamma() * self.opt_weights) if order >= 2 else 0
        speed = -np.sum(greeks.speed() * self.opt_weights) if order >= 3 else 0

        first_order = self._get_first_order(delta, gamma, speed)
        second_order = self._get_second_order(delta, gamma, speed)
        third_order = self._get_third_order(delta, gamma, speed)
        fourth_order = self._get_fourth_order(delta, gamma, speed)

        variance = second_order - first_order**2
        kurtosis = ((fourth_order - 4 * third_order * first_order + 6 * second_order * first_order **2 - 3 * first_order **4)/np.sqrt(variance) **4) - 3
        skewness = (third_order - 3 * second_order * first_order + 2 * first_order ** 3) / np.sqrt(variance) **3

        z_a = norm.ppf(alpha)
        z_tilde = z_a + (z_a **2 - 1) * skewness / 6

        # Conditions vectorisées pour skewness et kurtosis
        mask_skewinf = np.abs(skewness) <= 6 * (np.sqrt(2) - 1)
        mask_skewsup = np.abs(skewness) >= 6 * (np.sqrt(2) + 1)

        # Pour les valeurs qui satisfont ces conditions, applique l'ajustement
        s = skewness / 6
        k = kurtosis / 24

        # Calcul du z_tilde avec ajustement en utilisant un masque pour éviter les ifs
        k_p = (1 + 11 * s ** 2 - np.sqrt(s ** 4 - 6 * s ** 2 + 1)) / 6
        k_p2 = (1 + 11 * s ** 2 + np.sqrt(s ** 4 - 6 * s ** 2 + 1)) / 6

        # Conditions d'acceptation de k, k_p, k_p2 en vectorisé
        z_tilde = np.where(
            (mask_skewinf | mask_skewsup & (k_p <= k) & (k <= k_p2)),
            z_a + (z_a ** 2 - 1) * s + (z_a ** 3 - 3 * z_a) * k - (
                        2 * z_a ** 3 - 5 * z_a) * skewness ** 2 / 36,
            z_tilde  # Si les conditions ne sont pas remplies, on garde simplement z_a (pas de correction)
        )

        # Résultat final avec z_tilde ajusté
        return np.sum(- (first_order + np.sqrt(variance) * z_tilde))


    def _get_first_order(self, delta: float, gamma: float, speed: float):
        return 1/2 * gamma * self.S0 **2 * self.sigma_var **2
    
    def _get_second_order(self, delta: float, gamma: float, speed: float):
        return  (1/36 * speed **2 * self.S0 **6) * 15 * self.sigma_var **6 + \
                ((1/4 * gamma **2 * self.S0 **4)+(1/3 * speed*self.S0 **3)*(delta * self.S0))* \
                3 * (self.sigma_var) **4 + \
                (delta **2 * self.S0 **2) * (self.sigma_var) **2

    
    def _get_third_order(self, delta: float, gamma: float, speed: float):
        return (3/2 * gamma * self.S0**2) * (1/36 * speed**2 * self.S0**6) * 105 * self.sigma_var**8 + \
                ((1/8 * gamma**3 * self.S0**6) + (speed * self.S0**3) * (1/2 * gamma * self.S0**2) *
                (delta * self.S0)) * 15 * self.sigma_var**6 + \
                (3/2 * gamma * self.S0**2) * \
                (delta**2 * self.S0**2) * 3 * self.sigma_var**4


    def _get_fourth_order(self, delta: float, gamma: float, speed: float):
        return ((1/1296) * speed**4 * self.S0**12) * 10395 * self.sigma_var**12 + \
                ((1/6 * speed**2 * self.S0**6) * (1/4 * gamma**2 * self.S0**4) + 4 * (delta * self.S0) *
                (1/216 * speed**3 * self.S0**9)) * 945 * self.sigma_var**10 + \
                ((1/16 * gamma**4 * self.S0**8) + (2 * speed * self.S0**3) * (delta * self.S0) *
                (1/4 * gamma**2 * self.S0**4) + (1/6 * speed**2 * self.S0**6) * (delta**2 * self.S0**2)) * \
                105 * self.sigma_var**8 + \
                ((3/2 * gamma**2 * self.S0**4) * (delta**2 * self.S0**2) + (2/3 * speed * self.S0**3) *
                (delta**3 * self.S0**3)) * 15 * self.sigma_var**6 + \
                (delta**4 * self.S0**4) * 3 * self.sigma_var**4


if __name__ == "__main__":
    from option.option import Call, Put, OptionPortfolio
    from investment_strategies.vanilla_strategies import Straddle
    from pricers.bs_pricer import BSPortfolio
    from pricers.tree_pricer import TreePortfolio
    from pricers.mc_pricer import MonteCarloEngine

    from market.market_factory import create_market

    # === 1) Définir la date de pricing et la maturité (5 ans) ===
    pricing_date = datetime(2023, 4, 25)
    maturity_date = datetime(2028, 4, 25)

    # === 2) Paramètres pour Svensson ===
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]
    # === 3) Instanciation « tout‐en‐un » du Market LVMH ===
    market = create_market(
        stock="LVMH",
        pricing_date=pricing_date,
        vol_source="implied",  # ou "historical"
        hist_window=252,
        curve_method="svensson",  # méthode de calibration
        curve_kwargs={"initial_guess": sv_guess},
        dcc="Actual/Actual",
    )

    K = market.S0 * 0.9

    call_option = Call(K=K, maturity=maturity_date)
    put_option = Put(K=K, maturity=maturity_date)
    opt_ptf = OptionPortfolio([call_option, put_option]) # Portefeuille quelconque

    shift_date = pricing_date + timedelta(days=30) # VaR à 30 jours

    # === Backtest Setup (Black Scholes) ===
    print(f'\n -------- VaR via modèle de Black Scholes --------')
    bse = BSPortfolio(market=market, option_ptf=opt_ptf, pricing_date=pricing_date)
    backtest = Backtest(model=bse, shift_date=shift_date)

    # === VaR Théorique ===
    var_th = backtest.run(var_type="TH", alpha=0.05)
    print(f"VaR Théorique du Portefeuille : {var_th}")

    # === VaR Monte Carlo ===
    var_mc = backtest.run(var_type="MC", alpha=0.05, nb_simu=1000)
    print(f"VaR Monte Carlo du Portefeuille : {var_mc}")

    # === VaR Cornish Fisher ===
    var_cf = backtest.run(var_type="CF", alpha=0.05, order=2)
    print(f"VaR Cornish Fisher du Portefeuille : {var_cf}")

    # Vérification par stratégie vanille
    straddle_ptf = Straddle(strike=K, pricing_date=pricing_date, maturity_date=maturity_date)
    opt_ptf = OptionPortfolio(options=straddle_ptf.options, weights=straddle_ptf.weights)

    # === Backtest Setup (Black Scholes) ===
    bse = BSPortfolio(market=market, option_ptf=opt_ptf, pricing_date=pricing_date)
    backtest = Backtest(model=bse, shift_date=shift_date)

    # === VaR Théorique ===
    var_th = backtest.run(var_type="TH", alpha=0.05)
    print(f"VaR Théorique du Straddle : {var_th}")

    # === VaR Monte Carlo ===
    var_mc = backtest.run(var_type="MC", alpha=0.05, nb_simu=100)
    print(f"VaR Monte Carlo du Straddle : {var_mc}")

    # === VaR Cornish Fisher ===
    var_cf = backtest.run(var_type="CF", alpha=0.05, order=2)
    print(f"VaR Cornish Fisher du Straddle: {var_cf}")

    # === Backtest Setup (Trinomial) ===
    print(f'\n --------  VaR via modèle Trinomial -------- ')
    te = TreePortfolio(market=market, option_ptf=opt_ptf, pricing_date=pricing_date, n_steps=300)
    backtest = Backtest(model=te, shift_date=shift_date)

    # === VaR Théorique ===
    var_th = backtest.run(var_type="TH", alpha=0.05)
    print(f"VaR Théorique du Portefeuille : {var_th}")

    # === VaR Monte Carlo ===
    var_mc = backtest.run(var_type="MC", alpha=0.05, nb_simu=100)
    print(f"VaR Monte Carlo du Portefeuille : {var_mc}")

    # === VaR Cornish Fisher ===
    var_cf = backtest.run(var_type="CF", alpha=0.05, order=2)
    print(f"VaR Cornish Fisher du Portefeuille : {var_cf}")

    # === Backtest Setup (Monte Carlo) ===
    print(f'\n --------  VaR via modèle Monte Carlo -------- ')
    mce = MonteCarloEngine(market=market, option_ptf=opt_ptf, pricing_date=pricing_date, n_paths=10000, n_steps=300, seed=2)
    backtest = Backtest(model=mce, shift_date=shift_date)

    # === VaR Théorique ===
    var_th = backtest.run(var_type="TH", alpha=0.05)
    print(f"VaR Théorique du Portefeuille : {var_th}")

    # === VaR Monte Carlo ===
    var_mc = backtest.run(var_type="MC", alpha=0.05, nb_simu=100)
    print(f"VaR Monte Carlo du Portefeuille : {var_mc}")

    # === VaR Cornish Fisher ===
    var_cf = backtest.run(var_type="CF", alpha=0.05, order=2)
    print(f"VaR Cornish Fisher du Portefeuille : {var_cf}")



