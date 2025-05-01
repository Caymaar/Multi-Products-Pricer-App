import configparser
import pandas as pd
from pathlib import Path
from typing import Union
from datetime import datetime, timedelta
from market.day_count_convention import DayCountConvention

def _find_project_root() -> Path:
    """
    Remonte l'arborescence jusqu'à trouver config/config.ini.
    """
    f = Path(__file__).resolve()
    for parent in f.parents:
        if (parent / "config" / "config.ini").is_file():
            return parent
    raise FileNotFoundError("Impossible de trouver config/config.ini dans les parents de " +
                             str(f))

def get_config() -> configparser.ConfigParser:
    project_root = _find_project_root()
    config = configparser.ConfigParser()
    config_path = project_root / "config" / "config.ini"
    config.read(config_path)
    return config

def get_price_data(stock_name: str) -> pd.Series:
    cfg = get_config()
    project_root = _find_project_root()

    prices_path = project_root / cfg['paths']['data_path'] / cfg['paths']['prices_path']
    section = f"stocks.{stock_name.upper()}"
    if section not in cfg:
        raise ValueError(f"Stock name '{stock_name}' not found in config. Sections disponibles : "
                         f"{cfg.sections()}")
    file_path = prices_path / f"{stock_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier de prix introuvable : {file_path}")
    #df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    #df.set_index('Date', inplace=True)
    #return df.sort_index().squeeze()
    # Charger et retourner la data
    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()  # Assurez-vous que les dates sont triées
    return df.squeeze()

def get_rate_data(rate_name: str, type: str) -> Union[pd.DataFrame, pd.Series]:
    cfg = get_config()
    project_root = _find_project_root()

    rate_path = project_root / cfg['paths']['data_path'] / cfg['paths'][f'{type.lower()}_rate_path']
    file_path = rate_path / f"{rate_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier de taux introuvable : {file_path}")
    df = pd.read_csv(file_path, sep=';', parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    return df.squeeze() if df.shape[1] == 1 else df

def get_zone(stock_name: str) -> str:
    cfg = get_config()
    section = f"stocks.{stock_name.upper()}"
    if section not in cfg:
        raise ValueError(f"Stock name '{stock_name}' not found in config. Sections disponibles : "
                         f"{cfg.sections()}")
    return cfg[section]['zone']

def get_implied_vol(stock_name: str) -> pd.DataFrame:
    cfg = get_config()
    project_root = _find_project_root()

    implied_path = project_root / cfg['paths']['data_path'] / cfg['paths']['options_path']
    file_path = implied_path / f"{stock_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier d’implied vol introuvable : {file_path}")
    return pd.read_csv(file_path, sep=';')

def tenor_to_years(tenor: str, start_date: datetime.date = None, dcc: str = "Actual/365") -> float:
    from dateutil.relativedelta import relativedelta

    unit = tenor[-1].upper()
    n = int(tenor[:-1])
    dcc_obj = DayCountConvention(dcc)
    if start_date is not None:
        if unit == 'W':
            end = start_date + timedelta(weeks=n)
        elif unit == 'M':
            end = start_date + relativedelta(months=n)
        elif unit == 'Y':
            end = start_date + relativedelta(years=n)
        else:
            raise ValueError(f"Unité de tenor inconnue : {unit}")
        return dcc_obj.year_fraction(start_date, end)

    # sans date, approche implicite
    days_in_year = 360 if "360" in dcc.lower() else 365
    if unit == 'W':
        return n * 7 / days_in_year
    if unit == 'M':
        return n * 30 / days_in_year
    if unit == 'Y':
        return float(n)
    raise ValueError(f"Unité de tenor inconnue : {unit}")
