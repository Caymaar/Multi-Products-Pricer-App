from pricers.mc_pricer import MonteCarloEngine
from rate.curve_utils import make_zc_curve
from rate.products import ZeroCouponBond
from option.option import Option
from market.market import Market
from datetime import datetime
from investment_strategies.structured_strategies import StructuredProduct

class StructuredPricer:
    def __init__(self,
                 zc_method: str,
                 zc_args: tuple,
                 market: Market,
                 pricing_date: datetime,
                 n_paths: int = 100000,
                 n_steps: int = 300,
                 seed: int = None):
        self.zc_curve = make_zc_curve(zc_method, *zc_args)
        self.market = market
        self.pricing_date = pricing_date
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def price(self, product: StructuredProduct, engine_type="MC") -> float:
        legs = product.get_legs()

        # 1) ZCB de protection
        zcb_leg, sign_zcb = next(
            ((l, s) for l, s in legs if isinstance(l, ZeroCouponBond)),
            (None, 0.0)
        )
        p_zcb = zcb_leg.price(self.zc_curve)
        total = sign_zcb * p_zcb

        # 2) résiduel à investir
        residu = product.notional - sign_zcb * p_zcb

        # 3) collecte des legs optionnels
        opt_legs = [(opt, sign) for opt, sign in legs if isinstance(opt, Option)]
        m = len(opt_legs)
        if m == 0 or residu <= 0:
            return total

        # 4) pricing unitaire des options
        #    et allocation équipondérée du résiduel
        for opt, sign in opt_legs:
            engine = MonteCarloEngine(
                market=self.market,
                option=opt,
                pricing_date=self.pricing_date,
                n_paths=self.n_paths,
                n_steps=self.n_steps,
                seed=self.seed
            )
            p_opt = engine.price(type=engine_type)
            # qty fractionnaire pour tout investir en parts égales
            qty = sign * (residu / m) / p_opt if p_opt > 0 else 0.0
            total += qty * p_opt

        # NOTE : residu/m * m == residu, donc aucun cash laissé de côté
        #       (on a investi 100% du résiduel en parts égales)
        return total