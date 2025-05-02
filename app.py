from enum import Enum
from datetime import datetime, timedelta

from investment_strategies.structured_strategies import (
    ReverseConvertible,
    TwinWin,
    BonusCertificate,
    CappedParticipationCertificate,
    DiscountCertificate,
    ReverseConvertibleBarrier,
    SweetAutocall
)

from option.option import (
    Call, Put,
    DigitalCall, DigitalPut,
    UpAndOutCall, DownAndOutPut,
    UpAndInCall, DownAndInPut,
    UpAndOutPut, DownAndOutCall,
    UpAndInPut, DownAndInCall
)

from investment_strategies.vanilla_strategies import (
    BearCallSpread,
    BullCallSpread,
    ButterflySpread,
    Straddle,
    Strap,
    Strip,
    Strangle,
    Condor,
    PutCallSpread
)

from rate.products import (
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    ForwardRateAgreement,
    InterestRateSwap
)

from risk_metrics.rate_product_sensitivity import (
    ZeroCouponSensitivity,
    FixedRateBondSensitivity,
    FloatingRateBondSensitivity,
    InterestRateSwapSensitivity,
)

class Options(Enum):
    Call          = Call
    Put           = Put
    DigitalCall     = DigitalCall
    DigitalPut      = DigitalPut
    UpAndOutCall    = UpAndOutCall
    DownAndOutCall  = DownAndOutCall
    UpAndOutPut     = UpAndOutPut
    DownAndOutPut   = DownAndOutPut
    UpAndInCall     = UpAndInCall
    DownAndInCall   = DownAndInCall
    UpAndInPut      = UpAndInPut
    DownAndInPut    = DownAndInPut

class Structured(Enum):
    ReverseConvertible             = ReverseConvertible
    TwinWin                        = TwinWin
    BonusCertificate               = BonusCertificate
    CappedParticipationCertificate = CappedParticipationCertificate
    DiscountCertificate            = DiscountCertificate
    ReverseConvertibleBarrier      = ReverseConvertibleBarrier
    SweetAutocall                  = SweetAutocall

class Strategy(Enum):
    BearCallSpread  = BearCallSpread
    BullCallSpread  = BullCallSpread
    ButterflySpread = ButterflySpread
    Straddle        = Straddle
    Strap           = Strap
    Strip           = Strip
    Strangle        = Strangle
    Condor          = Condor
    PutCallSpread   = PutCallSpread

class Rate(Enum):
    ZeroCouponBond       = ZeroCouponBond
    FixedRateBond        = FixedRateBond
    FloatingRateBond     = FloatingRateBond
    ForwardRateAgreement = ForwardRateAgreement
    InterestRateSwap     = InterestRateSwap

class Sensitivity(Enum):
    ZeroCouponBond       = ZeroCouponSensitivity
    FixedRateBond        = FixedRateBondSensitivity
    FloatingRateBond     = FloatingRateBondSensitivity
    InterestRateSwap     = InterestRateSwapSensitivity

class Category(Enum):
    OPTION     = Options
    STRATEGY   = Strategy
    STRUCTURED = Structured
    RATE       = Rate

# OptionStrategies Enum
class OptionStrategies(Enum):
    Call         = Call
    Put          = Put
    BearCallSpread = BearCallSpread
    PutCallSpread  = PutCallSpread
    BullCallSpread = BullCallSpread
    ButterflySpread= ButterflySpread
    Straddle       = Straddle
    Strap          = Strap
    Strip          = Strip
    Strangle       = Strangle
    Condor         = Condor
    DigitalCall     = DigitalCall
    DigitalPut      = DigitalPut
    UpAndOutCall    = UpAndOutCall
    DownAndOutCall  = DownAndOutCall
    UpAndOutPut     = UpAndOutPut
    DownAndOutPut   = DownAndOutPut
    UpAndInCall     = UpAndInCall
    DownAndInCall   = DownAndInCall
    UpAndInPut      = UpAndInPut
    DownAndInPut    = DownAndInPut

COMMON_PARAMS = {
Category.OPTION: [
        {"name": "K", "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}},
        {"name": "maturity", "widget": "date_input", "label": "Maturité", "kwargs": {"value_func": lambda today: datetime(2023, 1, 1) + timedelta(days=365)}},
        {"name": "exercise", "widget": "selectbox", "label": "Exercice", "kwargs": {"options": ["european", "american"], "index": 0}},
    ],
Category.STRATEGY: [
        {"name": "maturity_date", "widget": "date_input", "label": "Maturité", "kwargs": {"value_func": lambda today: datetime(2023, 1, 1) + timedelta(days=365)}},
    ],
Category.STRUCTURED: [
        {"name": "maturity_date", "widget": "date_input", "label": "Maturité", "kwargs": {"value_func": lambda today: datetime(2023, 1, 1) + timedelta(days=365)}},
        {"name": "notional", "widget": "number_input", "label": "Nominal", "kwargs": {"value": 1000.0}},
    ],
Category.RATE: [
    ],
}

SPECIFIC_PARAMS = {
    Category.OPTION: {
        "Call": [],
        "Put": [],
        "DigitalCall": [
            {"name": "payoff", "widget": "number_input", "label": "Payoff", "kwargs": {"value": 1.0}}
        ],
        "DigitalPut": [
            {"name": "payoff", "widget": "number_input", "label": "Payoff", "kwargs": {"value": 1.0}}
        ],
        "UpAndOutCall": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 110.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ],
        "DownAndOutCall": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 90.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ],
        "UpAndOutPut": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 110.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ],
        "DownAndOutPut": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 90.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ],
        "UpAndInCall": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 110.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ],
        "DownAndInCall": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 90.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ],
        "UpAndInPut": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 110.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ],
        "DownAndInPut": [
            {"name": "barrier", "widget": "number_input", "label": "Barrier", "kwargs": {"value": 90.0}},
            {"name": "rebate",  "widget": "number_input", "label": "Rebate",  "kwargs": {"value": 0.0}}
        ]
    },
    Category.STRATEGY: {
        "BearCallSpread": [
            {"name": "strike_sell", "widget": "number_input", "label": "Strike Sell (%)", "kwargs": {"value": 95.0}},
            {"name": "strike_buy",  "widget": "number_input", "label": "Strike Buy (%)", "kwargs": {"value": 105.0}}
        ],
        "PutCallSpread": [
            {"name": "strike", "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}}
        ],
        "BullCallSpread": [
            {"name": "strike_buy", "widget": "number_input", "label": "Strike Buy (%)", "kwargs": {"value": 95.0}},
            {"name": "strike_sell", "widget": "number_input", "label": "Strike Sell (%)", "kwargs": {"value": 105.0}}
        ],
        "ButterflySpread": [
            {"name": "strike_low", "widget": "number_input", "label": "Strike Low (%)", "kwargs": {"value": 90.0}},
            {"name": "strike_mid", "widget": "number_input", "label": "Strike Mid (%)", "kwargs": {"value": 100.0}},
            {"name": "strike_high", "widget": "number_input", "label": "Strike High (%)", "kwargs": {"value": 110.0}}
        ],
        "Straddle": [
            {"name": "strike", "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}}
        ],
        "Strap": [
            {"name": "strike", "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}}
        ],
        "Strip": [
            {"name": "strike", "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}}
        ],
        "Strangle": [
            {"name": "lower_strike", "widget": "number_input", "label": "Lower Strike (%)", "kwargs": {"value": 90.0}},
            {"name": "upper_strike", "widget": "number_input", "label": "Upper Strike (%)", "kwargs": {"value": 110.0}}
        ],
        "Condor": [
            {"name": "strike1", "widget": "number_input", "label": "Strike 1 (%)", "kwargs": {"value": 90.0}},
            {"name": "strike2", "widget": "number_input", "label": "Strike 2 (%)", "kwargs": {"value": 95.0}},
            {"name": "strike3", "widget": "number_input", "label": "Strike 3 (%)", "kwargs": {"value": 105.0}},
            {"name": "strike4", "widget": "number_input", "label": "Strike 4 (%)", "kwargs": {"value": 110.0}}
        ],
    },
    Category.STRUCTURED: {
        "ReverseConvertible": [
            {"name": "K", "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}}
        ],
        "TwinWin": [
            {"name": "K",           "widget": "number_input", "label": "Strike (%)",         "kwargs": {"value": 100.0}},
            {"name": "PDO_barrier", "widget": "number_input", "label": "PDO Barrier (↓)",    "kwargs": {"value": 80.0}},
            {"name": "CUO_barrier", "widget": "number_input", "label": "CUO Barrier (↑)",    "kwargs": {"value": 120.0}}
        ],
        "SweetAutocall": [
            {"name": "frequency",          "widget": "selectbox", "label": "Fréquence d'observation", "kwargs": {"options": ["Annuel", "Semestriel", "Trimestriel"], "index": 2}},
            {"name": "coupon_rate",        "widget": "number_input", "label": "Coupon Rate (%)",        "kwargs": {"value": 5.00}},
            {"name": "coupon_barrier",     "widget": "number_input", "label": "Coupon Barrier (%)",     "kwargs": {"value": 80.0}},
            {"name": "call_barrier",       "widget": "number_input", "label": "Call Barrier (%)",       "kwargs": {"value": 110.0}},
            {"name": "protection_barrier", "widget": "number_input", "label": "Protection Barrier (%)", "kwargs": {"value": 80.0}},
        ],
        "BonusCertificate": [
            {"name": "K",        "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}},
            {"name": "barrier",  "widget": "number_input", "label": "Barrier (%)",     "kwargs": {"value": 80.0}}
        ],
        "CappedParticipationCertificate": [
            {"name": "K",   "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}},
            {"name": "cap", "widget": "number_input", "label": "Cap (%)",         "kwargs": {"value": 120.0}}
        ],
        "DiscountCertificate": [
            {"name": "K", "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}}
        ],
        "ReverseConvertibleBarrier": [
            {"name": "K",       "widget": "number_input", "label": "Strike (%)", "kwargs": {"value": 100.0}},
            {"name": "barrier", "widget": "number_input", "label": "Barrier (%)",     "kwargs": {"value": 80.0}}
        ]
    },
    Category.RATE : {
    "ZeroCouponBond": [
        {"name": "face_value", "widget": "number_input", "label": "Nominal", "kwargs": {"value": 1000.0}},
        {"name": "maturity_date", "widget": "date_input", "label": "Maturité",
         "kwargs": {"value_func": lambda today: today + timedelta(days=365)}},
    ],
    "FixedRateBond": [
        {"name": "face_value", "widget": "number_input", "label": "Nominal", "kwargs": {"value": 1000.0}},
        {"name": "maturity_date", "widget": "date_input", "label": "Maturité",
         "kwargs": {"value_func": lambda today: today + timedelta(days=365)}},
        {"name": "coupon_rate", "widget": "number_input", "label": "Taux de coupon", "kwargs": {"value": 5.00}},
        {"name": "frequency", "widget": "selectbox", "label": "Fréquence", "kwargs": {
            "options": ["Annuel", "Semestriel", "Trimestriel"],
            "index": 1
        }},
    ],
    "FloatingRateBond": [
        {"name": "face_value", "widget": "number_input", "label": "Nominal", "kwargs": {"value": 1000.0}},
        {"name": "maturity_date", "widget": "date_input", "label": "Maturité", "kwargs": {"value_func": lambda today: today + timedelta(days=365)}},
        {"name": "margin", "widget": "number_input", "label": "Margin (s)", "kwargs": {"value": 0.00}},
        {"name": "frequency", "widget": "selectbox", "label": "Fréquence", "kwargs": {"options": ["Annuel", "Semestriel", "Trimestriel"],"index": 1}},
        {"name": "multiplier", "widget": "number_input", "label": "Multiplier (M)", "kwargs": {"value": 1.00}},
    ],
    "ForwardRateAgreement": [
        {"name": "notional", "widget": "number_input", "label": "Nominal", "kwargs": {"value": 1_000_000.0}},
        {"name": "start_date", "widget": "date_input", "label": "Start FRA","kwargs": {"value_func": lambda today: today + timedelta(days=30)}},
        {"name": "end_date", "widget": "date_input", "label": "End FRA","kwargs": {"value_func": lambda today: today + timedelta(days=120)}},
        {"name": "reference_rate", "widget": "selectbox", "label": "Reference", "kwargs": {"options": ["Overnight", "3M"],"index": 1}},
    ],
    "InterestRateSwap": [
        {"name": "notional", "widget": "number_input", "label": "Nominal", "kwargs": {"value": 1_000_000.0}},
        {"name": "start_date", "widget": "date_input", "label": "Date de départ", "kwargs": {"value_func": lambda today: today + timedelta(days=2)}},
        {"name": "end_date", "widget": "date_input", "label": "Date de maturité","kwargs": {"value_func": lambda today: today + timedelta(days=365 * 5)}},
        {"name": "frequency", "widget": "selectbox", "label": "Fréquence", "kwargs": {"options": ["Annuel", "Semestriel", "Trimestriel"],"index": 1}},
        {"name": "spread", "widget": "number_input", "label": "Margin (s)", "kwargs": {"value": 0.00}},
        {"name": "multiplier", "widget": "number_input", "label": "Multiplier (M)", "kwargs": {"value": 1.00}},
    ]}
}