import pandas as pd

class OptionDataParser:
    @staticmethod
    def parse_option_row(row, ref_date=None):
        """
        Extrait la date d'échéance, le strike et la volatilité implicite d'une ligne de DataFrame.
        La fonction se base sur la colonne 'Ticker' pour extraire la date et le strike.

        :param row: une ligne du DataFrame
        :param ref_date: date de référence pour calculer le temps à expiration (par défaut, aujourd'hui)
        :return: (time_to_expiry, strike, vol)
        """
        # Utilise la date d'aujourd'hui comme référence si non précisée
        if ref_date is None:
            ref_date = pd.Timestamp.today()

        # Extraction depuis la colonne 'Ticker'
        # Exemple de format attendu : "AAPL 3/21/25 C207.5"
        ticker_str = row['Ticker']
        try:
            tokens = ticker_str.split()
            # La deuxième valeur correspond à la date d'échéance
            expiry_str = tokens[1]
            # La troisième valeur contient le type et le strike (ex: "C207.5")
            option_info = tokens[2]
            # On extrait le strike en enlevant la première lettre
            strike = float(option_info[1:])

            # Conversion de la date d'échéance en datetime (format supposé mm/dd/yy)
            expiry_date = pd.to_datetime(expiry_str, format='%m/%d/%y')
            # Calcul du temps jusqu'à expiration en années
            time_to_expiry = (expiry_date - ref_date).days / 365.0

            # Récupération de la volatilité implicite : on essaye 'VIM' puis 'VIM.1'
            vol = row.get('VIM')
            if pd.isna(vol):
                vol = row.get('VIM.1')
            # On suppose que vol est déjà en format numérique (sinon conversion nécessaire)
            return time_to_expiry, strike, vol
        except Exception as e:
            print(f"Erreur lors du parsing de la ligne : {ticker_str} - {e}")
            return None, None, None

    @staticmethod
    def prepare_option_data(file_path):
        """
        Lit le fichier Excel et prépare un DataFrame avec les colonnes nécessaires :
        'maturity' (temps en années), 'strike' et 'vol'.
        On ignore la première ligne de données si elle contient des informations d'en-tête.

        :param file_path: chemin du fichier Excel
        :return: DataFrame avec colonnes ['maturity', 'strike', 'vol']
        """
        # Lecture en indiquant que la deuxième ligne contient les noms de colonnes
        data = pd.read_excel(file_path, header=1)

        # Optionnel : filtrer sur une option (par exemple, uniquement sur AAPL)
        # data = data[data['Ticker'].str.contains("AAPL", na=False)]

        # Initialisation d'une liste pour stocker les données extraites
        rows = []
        # Parcours des lignes
        for idx, row in data.iterrows():
            # On ignore les lignes sans valeur dans 'Ticker' ou qui ne sont pas des chaînes
            if pd.isna(row['Ticker']) or not isinstance(row['Ticker'], str):
                continue
            t_exp, strike, vol = OptionDataParser.parse_option_row(row)
            if t_exp is not None and strike is not None and vol is not None:
                rows.append({'maturity': t_exp, 'strike': strike, 'vol': vol})

        df_options = pd.DataFrame(rows)
        # Optionnel : filtrer pour garder uniquement les maturités positives
        df_options = df_options[df_options['maturity'] > 0]
        return df_options