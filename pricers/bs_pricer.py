from market.market import Market
from option.option import Call, Put, Option, OptionPortfolio
import numpy as np
import scipy.stats as stats
from datetime import datetime


class BSPortfolio:
    def __init__(self, market, option_ptf: OptionPortfolio, pricing_date):
        """
        :param market: Instance du marché
        :param option_ptf: Instance de OptionPortfolio
        :param pricing_date: Date de valorisation
        """
        self.market = market
        self.options = option_ptf
        self.pricing_date = pricing_date
        self.bse = {}  # Dictionnaire {(T, dt): TreeModel}
        self.T = np.array([self.market.dcc.year_fraction(start_date=pricing_date, end_date=option.T)
                           for option in self.options.assets])

        self._build_bse()

    def _build_bse(self):
        """
        Construit un BS Engine pour chaque option.
        """

        for option in self.options.assets:
            # Instancier un bse pour une option
            pricer = BlackScholesPricer(
                market=self.market,
                option=option,
                pricing_date=self.pricing_date
            )

            self.bse[option] = pricer

    def price(self, type=None):
        """
        Renvoi un vecteur ou float de prix de tous les groupes d'options.
        """
        prices = np.array([])
        for opt in self.bse.keys():
            prices = np.append(prices, self.bse[opt].price())
        return prices[-1] if len(prices)==1 else prices

    def _vectorized_greek(self, greek_name):
        greeks = np.array([])
        for opt in self.bse.keys():
            greek_func = getattr(self.bse[opt], greek_name)
            greeks = np.append(greeks, greek_func())
        greeks = np.array(greeks)
        return greeks[-1] if len(greeks) == 1 else greeks

    def delta(self):
        return self._vectorized_greek('delta')

    def gamma(self):
        return self._vectorized_greek('gamma')

    def vega(self):
        return self._vectorized_greek('vega')

    def theta(self):
        return self._vectorized_greek('theta')

    def rho(self):
        return self._vectorized_greek('rho')

    def speed(self):
        return self._vectorized_greek('speed')

    def recreate_model(self, **kwargs) -> "BSPortfolio":
        """
        Recrée une nouvelle instance du BlackScholesPricer en reprenant
        tous les paramètres actuels, sauf ceux passés en kwargs.
        """

        base_params = {
            "market":  self.market.copy(),
            "option_ptf":  self.options,
            "pricing_date":  self.pricing_date,
        }

        # Surcharge avec ce que l'utilisateur fournit
        base_params.update(kwargs)

        # Création de la nouvelle instance
        return BSPortfolio(**base_params)


# ---------------- Black-Scholes Pricer ----------------
class BlackScholesPricer:
    def __init__(self, market: Market, option: Option, pricing_date: datetime):
        self.market = market  # Informations sur le marché (spot, taux, etc.)
        self.pricing_date = pricing_date
        self.option = option  # Option (call ou put)
        self.T = self._calculate_T()
        self.t_div = self._calculate_t_div()
        self.S0 = self._adjust_initial_price()
        self.q = self._compute_dividend_yield()
        self.d1, self.d2 = None, None  # Initialisation des valeurs de d1 et d2

    def _calculate_T(self):
        """
        Méthode pour calculer les temps jusqu'à l'expiration des options (T).
        Retourne un array des durées en années entre la date de pricing et les dates de maturité.
        """
        return self.market.dcc.year_fraction(start_date=self.pricing_date, end_date=self.option.T)

    def _calculate_t_div(self):
        """
        Calcule l'indice temporel pour le dividende si le dividende est discret.
        """
        if self.market.div_type == "discrete" and self.market.div_date is not None:
            return self.market.dcc.year_fraction(start_date=self.pricing_date,end_date=self.market.div_date) # Conversion en année
        else:
            return None

    def european_exercise_check(self):
        """Détermine si une option américaine peut être traitée comme une option européenne."""

        # Vérifie si c'est une option européenne
        if self.option.exercise.lower() == "european":
            return True

        # Vérifie les conditions d'équivalence pour un Call américain
        if isinstance(self.option, Call) and (not self.market.dividend) and self.market.zero_rate(self.T) > 0:
            return True

        # Vérifie les conditions d'équivalence pour un Call américain
        if isinstance(self.option, Call) and (not self.market.dividend) and self.market.zero_rate(self.T) > 0:
            return True

        # Vérifie les conditions d'équivalence pour un Put américain
        if isinstance(self.option, Put) and (not self.market.dividend) and self.market.zero_rate(self.T) < 0:
            return True

        # Sinon, ce n'est pas une option équivalente à une européenne
        return False

    def _adjust_initial_price(self):
        """ Ajuste le prix initial en fonction des dividendes. """
        if self.market.div_type == "discrete" and self.market.div_date is not None:
            return self.market.S0 - self.market.dividend * np.exp(-self.market.zero_rate(self.t_div) * self.t_div)
        return self.market.S0

    def _compute_dividend_yield(self):
        """ Calcule le taux de dividende en fonction du type de dividende. """
        return self.market.dividend if self.market.div_type == "continuous" else 0

    def _compute_d1_d2(self):
        """ Calcule d1 et d2 pour la formule de Black-Scholes. """
        sigma_sqrt_T = self.market.sigma * np.sqrt(self.T)
        self.d1 = (np.log(self.S0 / self.option.K) +
                   (self.market.zero_rate(self.T) - self.q + 0.5 * self.market.sigma ** 2) * self.T) / sigma_sqrt_T
        self.d2 = self.d1 - sigma_sqrt_T

    def price(self):
        """ Calcul du prix de l'option via Black-Scholes. """
        if self.european_exercise_check():
            self._compute_d1_d2()
            if isinstance(self.option, Call):
                return self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1) - \
                    self.option.K * np.exp(-self.market.zero_rate(self.T) * self.T) * stats.norm.cdf(self.d2)

            elif isinstance(self.option, Put):
                return self.option.K * np.exp(-self.market.zero_rate(self.T) * self.T) * stats.norm.cdf(-self.d2) - \
                    self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)

            else:
                return "NA"

        else:
            return "NA"

    # ---------------- Greek Calculations ----------------

    def delta(self):
        """ Calcul du Delta (sensibilité au spot). """
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()

        if isinstance(self.option, Call):
            return np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
        elif isinstance(self.option, Put):
            return np.exp(-self.q * self.T) * (stats.norm.cdf(self.d1) - 1)

    def gamma(self):
        """ Calcul du Gamma (convexité par rapport au spot). """
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()

        return np.exp(-self.q * self.T) * stats.norm.pdf(self.d1) / (self.S0 * self.market.sigma * np.sqrt(self.T))

    def vega(self):
        """ Calcul du Vega (sensibilité à la volatilité). """
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()

        return self.S0 * np.exp(-self.q * self.T) * stats.norm.pdf(self.d1) * np.sqrt(self.T) / 100

    def theta(self):
        """ Calcul du Theta (décroissance temporelle). """
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()

        first_term = - (self.S0 * self.market.sigma * np.exp(-self.q * self.T) * stats.norm.pdf(self.d1)) / (
                    2 * np.sqrt(self.T))
        if isinstance(self.option, Call):
            second_term = self.q * self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
            third_term = - self.market.zero_rate(self.T) * self.option.K * np.exp(-self.market.zero_rate(self.T) * self.T) * stats.norm.cdf(self.d2)
            return (first_term + second_term + third_term)/self.market.dcc.days_in_year
        elif isinstance(self.option, Put):
            second_term = self.q * self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)
            third_term = self.market.zero_rate(self.T) * self.option.K * np.exp(-self.market.zero_rate(self.T) * self.T) * stats.norm.cdf(-self.d2)
            return (first_term - second_term + third_term)/self.market.dcc.days_in_year

    def rho(self):
        """ Calcul du Rho (sensibilité aux taux d'intérêt). """
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()

        if isinstance(self.option, Call):
            return self.option.K * self.T * np.exp(-self.market.zero_rate(self.T) * self.T) * stats.norm.cdf(self.d2) / 100
        elif isinstance(self.option, Put):
            return -self.option.K * self.T * np.exp(-self.market.zero_rate(self.T) * self.T) * stats.norm.cdf(-self.d2) / 100

    def speed(self):
        """ Calcul du Speed (dérivée seconde du prix de l'option par rapport au spot). """
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()

        # Calcul du terme de base du Gamma
        gamma_term = np.exp(-self.q * self.T) * stats.norm.pdf(self.d1) / (
                    self.S0 * self.market.sigma * np.sqrt(self.T))

        # Calcul de la dérivée de Gamma par rapport à S0
        # Dérivée de la formule du Gamma par rapport à S0
        dGamma_dS = (-gamma_term / self.S0) * (1 - self.d1 / (self.S0 * self.market.sigma * np.sqrt(self.T)))

        return dGamma_dS


    def all_greeks(self):
        return [self.delta(),self.gamma(),self.vega(),self.theta(),self.rho(), self.gamma()]
