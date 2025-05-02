## Pricer de Produits Dérivés & Structurés

Ce projet met en place un **Pricer multi-catégories** pour les produits dérivés et structurés, développé en Python OOP et déployé sous forme d'application web via **Streamlit**. Il permet de calculer les prix, les greeks et les métriques de risque (DV01, Duration, Convexity) pour une large gamme de produits financiers.

---

### Fonctionnalités principales

* **Support multi-catégories** :

  * Produits structurés
  * Options (vanilla et portefeuilles d’options)
  * Stratégies d’options (straddles, strangles, butterflies, etc.)
  * Produits de taux (ZC bonds, FRA, IRS, etc.)

* **Méthodes de pricing** :

  * Monte Carlo avec antithétique
  * Arborescences binomiales/trinomiales (Tree)
  * Black & Scholes (BS) pour les options

* **Greeks & Sensibilités** :

  * Delta, Gamma, Vega, Theta, Rho, Speed
  * DV01, Duration, Convexité pour les produits de taux

* **Courbes de taux dynamiques**

  * Nelson–Siegel, Svensson, interpolation simple
  * Actual/360, Actual/365, 30/360, Actual/Actual

* **Interface utilisateur**

  * Sélection du marché (ticker, date de valorisation, source de volatilité, fenêtre historique)
  * Paramétrage intuitif des produits via widgets Streamlit
  * Affichage interactif des résultats (prix, greeks, graphiques de payoff)

---

## 🔧 Prérequis

* Python **3.11+**
* Gestionnaire de paquets `uv` (optionnel, remplace pip pour la sync des dépendances)

---

## Installation

### 1. Avec `uv`

```bash
# Installer uv si nécessaire
tpip install uv

# Synchroniser et installer toutes les dépendances du projet
uv sync
```

### 2. Avec `pip`

```bash
# Installer directement depuis le dépôt courant
pip install .
```

---

##  Lancement de l’application

Assurez-vous d'être dans le répertoire racine du projet, là où se trouve `streamlit_option_pricer.py`.

```bash
uv run streamlit run streamlit_option_pricer.py
```

Si vous n'utilisez pas `uv` :

```bash
streamlit run streamlit_option_pricer.py
```

---

## Utilisation & Paramétrage

1. **Marché & Sources** (barre latérale)

   * **Ticker** : symbole de l’actif sous-jacent (e.g. `LVMH`)
   * **Date de valorisation** : date de calcul des prix
   * **Vol source** : volatilité implicite ou historique
   * **Fenêtre historique** : nombre de jours pour calcul de l’historique
   * **Courbe & Day-count** : méthode et convention de calcul des taux

2. **Choix de la catégorie**

   * Naviguez entre les onglets : `STRUCTURED`, `OPTION`, `STRATEGY`, `RATE`
   * Sélectionnez le produit ou la stratégie désirée

3. **Paramétrage des produits**

   * Widgets dynamiques selon produits (strike, maturité, coupon, barrière…)
   * Valeurs par défaut basées sur la date de valorisation

4. **Sélection de la méthode** (pour options & stratégies)

   * **MC** : Monte Carlo
   * **Tree** : Arbre binomial/trinomial
   * **BS** : Black & Scholes

5. **Lancement du pricer**

   * Cliquez sur le bouton ▶️ pour afficher :

     * **Prix** à l’unité ou en pourcentage
     * **Greeks** (si applicable)
     * **Métriques de risque** (DV01, Duration, Convexité)
     * **Graphiques de payoff** (pour les stratégies)


