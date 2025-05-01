from datetime import datetime
from typing import Literal, Tuple, Callable, Any, Dict
import numpy as np

from data.management.data_retriever import DataRetriever
from rate.zc_curve import ZCFactory
from market.market import Market
from data.management.data_utils import get_config

def get_volatility(
    dr: DataRetriever,
    date: datetime,
    vol_source: Literal["implied","historical"] = "implied",
    hist_window: int = 252
) -> float:
    if vol_source == "implied":
        iv = dr.get_implied_volatility(date)
        if iv is not None:
            return iv
    # Autrement utilisation de la volatilité historique sur fenêtre de calcul
    rets = np.log(dr.prices / dr.prices.shift(1)).dropna()
    return rets.loc[:date].tail(hist_window).std() * np.sqrt(hist_window)

def get_dividend_info(stock: str) -> Tuple[float,str,datetime|None]:
    cfg = get_config()
    sec = f"stocks.{stock.upper()}"
    d = float(cfg[sec].get("dividend", 0.0))
    t = cfg[sec].get("div_type", "continuous")
    dd = cfg[sec].get("div_date", None)
    return d, t, (datetime.fromisoformat(dd) if dd else None)

def get_discount_forward_and_zero_curves(
    dr: DataRetriever,
    date: datetime,
    method: Literal["interpolation","nelson-siegel","svensson","vasicek"] = "interpolation",
    curve_kwargs: Dict[str, Any] | None = None,
    dcc: str = "Actual/365"
) -> Tuple[
     Callable[[float], float],        # DF(t)
     Callable[[float, float], float], # fwd(t1,t2)
     Callable[[float], float]         # zr(t)
]:
    curve_kwargs = (curve_kwargs or {}).copy()
    rfr = dr.get_risk_free_curve(date)
    fwd = dr.get_floating_curve(date)

    zcf = ZCFactory(risk_free_curve=rfr,
                    floating_curve = fwd,
                    dcc            = dcc)

    if method in ("nelson-siegel","svensson") and "initial_guess" not in curve_kwargs:
        curve_kwargs["initial_guess"] = (
            [0.02,-0.01,0.01,1.5]
            if method=="nelson-siegel"
            else [0.02,-0.01,0.01,0.005,1.5,3.5]
        )

    # 1) zero‐rate continu
    zero_rate = zcf.make_zc_curve(method=method, **curve_kwargs, dcc=dcc).yield_value
    # 2) discount factor continu
    discount = zcf.discount_curve(method=method, **curve_kwargs, dcc=dcc)
    # 3) forward discret implicite
    forward = zcf.forward_curve(method=method, **curve_kwargs, dcc=dcc)

    return discount, forward, zero_rate

def create_market(
    stock: str,
    pricing_date: datetime,
    vol_source: Literal["implied","historical"] = "implied",
    hist_window: int = 252,
    curve_method: Literal["interpolation","nelson-siegel","svensson","vasicek"] = "interpolation",
    curve_kwargs: Dict[str, Any] | None = None,
    dcc: str = "Actual/365",
    flat_rate: float | None = None
) -> Market:
    """
    Crée un Market en 4 étapes :
      1) DataRetriever
      2) volatilité
      3) dividende
      4) discount‐curve (via ZCFactory + curve_kwargs)
    """
    dr = DataRetriever(stock)

    # Spot & sigma
    S0    = dr.get_prices(pricing_date)
    sigma = get_volatility(dr, pricing_date, vol_source, hist_window)

    # Dividendes
    div, div_type, div_date = get_dividend_info(stock)

    # 3) préparer les 3 courbes
    if flat_rate is not None:
        # on wrappe le taux constant en trois callables
        zr_curve    = lambda t: flat_rate
        discount    = lambda t: np.exp(-flat_rate * t)
        forward     = lambda t1, t2: flat_rate
    else:
        # on bootstrappe normalement
        discount, forward, zr_curve = get_discount_forward_and_zero_curves(
            dr            = dr,
            date          = pricing_date,
            method        = curve_method,
            curve_kwargs  = curve_kwargs,
            dcc           = dcc
        )

    #4 obtenir la matrice de corrélation spot/taux
    corr_matrix = dr.get_correlation

    # 4) assembler le Market
    return Market(
        S0               = S0,
        sigma            = sigma,
        dividend         = div,
        div_type         = div_type,
        div_date         = div_date,
        day_count        = dcc,
        discount_curve   = discount,
        forward_curve    = forward,
        zero_rate_curve  = zr_curve,
        corr_matrix      = corr_matrix
    )