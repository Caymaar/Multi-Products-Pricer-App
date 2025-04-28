import configparser
import os
import pandas as pd
from pathlib import Path
from typing import Union

def get_config():
    # Obtenir le chemin absolu du répertoire racine du projet
    project_root = Path(__file__).parent
    
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'config', 'config.ini')
    config.read(config_path)
    return config

def get_price_data(stock_name: str) -> pd.Series:
    config = get_config()
    
    # Récupérer les chemins
    project_root = Path(__file__).parent
    prices_path = project_root / config['paths']['data_path'] / config['paths']['prices_path']
    
    # Trouver le ticker associé
    stock_section = f'stocks.{stock_name.upper()}'
    if stock_section not in config:
        raise ValueError(f"Stock name '{stock_name}' not found in config.")
    
    # Construire le chemin du fichier (en supposant fichier CSV nommé {ticker}.csv)
    file_path = prices_path / f"{stock_name}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Price file for ticker '{stock_name}' not found at {file_path}")
    
    # Charger et retourner la data
    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()  # Assurez-vous que les dates sont triées
    return df.squeeze()

def get_rate_data(rate_name: str, type: str) -> Union[pd.DataFrame, pd.Series]:
    config = get_config()
    
    # Récupérer les chemins
    project_root = Path(__file__).parent
    rate_path = project_root / config['paths']['data_path'] / config['paths'][f'{type.lower()}_rate_path']
    
    df = pd.read_csv(rate_path / f"{rate_name}.csv", sep=';', parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    
    df = df.sort_index()
    
    if len(df.columns) == 1:
        df = df.squeeze()

    return df

def get_zone(stock_name: str) -> str:
    config = get_config()
    
    # Trouver le ticker associé
    stock_section = f'stocks.{stock_name.upper()}'
    if stock_section not in config:
        raise ValueError(f"Stock name '{stock_name}' not found in config.")

    return config[stock_section]['zone']

def get_implied_vol(stock_name: str) -> pd.DataFrame:
    config = get_config()
    
    # Récupérer les chemins
    project_root = Path(__file__).parent
    implied_vol_path = project_root / config['paths']['data_path'] / config['paths']['options_path']
    
    df = pd.read_csv(implied_vol_path / f"{stock_name}.csv", sep=';')

    return df


def tenor_to_years(tenor: str) -> float:
    unit = tenor[-1]
    n    = float(tenor[:-1])
    if   unit == 'W': return n * 7/365
    elif unit == 'M': return n * 30/365
    elif unit == 'Y': return n
    else:  raise ValueError(f"Unité inconnue : {unit}")