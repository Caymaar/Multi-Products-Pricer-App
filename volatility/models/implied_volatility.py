from scipy.optimize import minimize_scalar
from risk_metrics.greeks import GreeksCalculator  # En supposant que votre module de Greeks est disponible
from pricers.bs_pricer import BlackScholesPricer
from pricers.mc_pricer import MonteCarloEngine
from market.market import Market
from option.option import Call
from datetime import datetime

class ImpliedVolatilityCalculator:
    """
    Classe pour le calcul de la volatilité implicite, fonctionnant avec tout pricer prenant
    la volatilité comme paramètre via le module `Market`.
    """

    def __init__(self, market_price: float, model, greeks_calculator: GreeksCalculator):
        self.market_price = market_price  # Prix de marché observé
        self.model = model  # Modèle de pricing (MC, Tree, BS...)
        self.greeks_calculator = greeks_calculator  # Calculateur de Greeks
        self._cached_prices = {}  # Cache des prix calculés

    def theoretical_price(self, sigma: float) -> float:
        """
        Calcule le prix théorique de l'option pour une volatilité donnée.

        :param sigma: Volatilité utilisée dans le modèle.
        :return: Prix théorique de l'option.
        """
        updated_market = self.model.market.copy(sigma=sigma)
        if isinstance(self.model, MonteCarloEngine):
            updated_model = self.model._recreate_model(market=updated_market)
        else:
            self.model.market = updated_market
            updated_model = self.model
        return updated_model.price()

    def f(self, sigma: float) -> float:
        """
        Fonction cible pour le calcul de l'IV :
        La différence entre le prix théorique et le prix de marché.

        :param sigma: Volatilité.
        :return: Différence de prix.
        """
        return self.theoretical_price(sigma) - self.market_price

    def vega(self, sigma: float) -> float:
        """
        Calcule le Vega via GreeksCalculator après réinstanciation du modèle avec la volatilité donnée.

        :param sigma: Volatilité pour laquelle calculer Vega.
        :return: Vega calculé.
        """
        updated_market = self.model.market.copy(sigma=sigma)
        updated_model = self.model._recreate_model(market=updated_market)
        self.greeks_calculator = GreeksCalculator(updated_model)  # Réinstancie le calculateur de Greeks
        return self.greeks_calculator.vega()

    def calculate_by_dichotomy(self, sigma_low=1e-5, sigma_high=5.0, tol=1e-7, max_iter=100) -> float:
        """
        Calcule la volatilité implicite par la méthode de dichotomie.

        :param sigma_low: Borne inférieure initiale.
        :param sigma_high: Borne supérieure initiale.
        :param tol: Tolérance d'arrêt.
        :param max_iter: Nombre maximal d'itérations.
        :return: Estimation de la volatilité implicite.
        """
        f_low = self.f(sigma_low)
        f_high = self.f(sigma_high)
        if f_low * f_high > 0:
            raise ValueError("Les bornes ne sont pas adaptées à la méthode de dichotomie.")

        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2.0
            f_mid = self.f(sigma_mid)
            if abs(f_mid) < tol:
                return sigma_mid
            if f_low * f_mid < 0:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
        return (sigma_low + sigma_high) / 2.0

    def calculate_by_optimization(self, bounds=(1e-5, 5.0), tol=1e-7, initial_guess=None) -> float:
        """
        Calcule la volatilité implicite en minimisant l'erreur quadratique entre le prix théorique
        et le prix de marché, en utilisant un point initial facultatif.

        :param bounds: Tuple (min, max) pour la recherche de la volatilité.
        :param initial_guess: Estimation initiale pour l'optimisation.
        :return: Estimation de la volatilité implicite.
        :raises Exception: Si l'optimisation échoue à converger.
        """

        def objective(sigma):
            return self.f(sigma) ** 2

        # Si aucun point initial n'est fourni, prendre le milieu des bornes
        if initial_guess is None:
            initial_guess = (bounds[0] + bounds[1]) / 2

        # Utiliser `minimize_scalar` avec les bornes et une estimation initiale
        result = minimize_scalar(
            objective,
            bounds=bounds,
            method='bounded',
            options = {'xatol': tol}
        )

        if result.success:
            return result.x
        else:
            raise Exception("L'optimisation n'a pas convergé.")


# Exemple d'utilisation avec un ensemble de données généré

if __name__ == "__main__":
    # ---------------- Génération de données de test ----------------
    pricing_date = datetime.strptime("2025-04-04", "%Y-%m-%d")
    maturity_date = datetime.strptime("2026-04-04", "%Y-%m-%d")

    # Marché et option
    market = Market(S0=100, r=0.05, sigma=0.2, dividend=0)
    option = Call(K=100, maturity=maturity_date)

    # Monte Carlo Model
    mc_model = MonteCarloEngine(market, option, pricing_date, n_paths=10000, n_steps=100, seed=2)

    # Calculateur de grecques (utile pour le vega pour N-R)
    GC = GreeksCalculator(mc_model)

    # Black-Scholes Pricer (pas besoin de GreeksCalculator)
    bs_pricer = BlackScholesPricer(market, option, t_div=0, dt=(maturity_date - pricing_date).days / 360, T=1.0)

    # Prix de marché fictif
    market_price = 10.0

    # ---------------- Test avec Monte Carlo Model ----------------
    print("----- Monte Carlo Model -----")
    mc_iv_calculator = ImpliedVolatilityCalculator(market_price, mc_model, GC)

    iv_mc_dicho = mc_iv_calculator.calculate_by_dichotomy()
    print(f"Volatilité implicite Monte Carlo (Dichotomie) : {iv_mc_dicho:.6f}")

    iv_mc_opt = mc_iv_calculator.calculate_by_optimization()
    print(f"Volatilité implicite Monte Carlo (Optimisation) : {iv_mc_opt:.6f}")

    # ---------------- Test avec Black-Scholes Pricer ----------------
    print("----- Black-Scholes Pricer -----")
    bs_iv_calculator = ImpliedVolatilityCalculator(market_price, bs_pricer, None)

    iv_bs_dicho = bs_iv_calculator.calculate_by_dichotomy()
    print(f"Volatilité implicite Black-Scholes (Dichotomie) : {iv_bs_dicho:.6f}")

    iv_bs_opt = bs_iv_calculator.calculate_by_optimization()
    print(f"Volatilité implicite Black-Scholes (Optimisation) : {iv_bs_opt:.6f}")
