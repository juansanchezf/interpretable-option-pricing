# Interpretable Neural Networks for Option Pricing

This repository contains all code, data, and models used in the development of my master’s thesis, which investigates the use of neural networks for financial option pricing. The work combines classical financial theory (Black-Scholes model) with modern machine learning (MLP architectures), and incorporates an interpretability analysis through sensitivity metrics and NeuralSens.

The repository is organized into four main parts:

- Data collection and preprocessing
- Model development and evaluation
- Sensitivity analysis and explainability
- Auxiliary experiments and synthetic datasets

## Repository Structure

```
.
├── data/               # All data used and generated during the experiments
│   ├── raw/            # Raw option chains downloaded from Yahoo Finance
│   ├── processed/      # Cleaned and preprocessed data after feature engineering
│   ├── predicted/      # Model predictions and sensitivity outputs
│   └── test/           # Synthetic datasets for controlled experiments
│
├── notebooks/          # Jupyter Notebooks for each stage of the pipeline
│   ├── 0. OptionScrapping.ipynb     # Automated data collection using yfinance
│   ├── 1. dataCleansing.ipynb       # Cleaning and feature engineering
│   ├── 2. EDA.ipynb                 # Exploratory data analysis
│   ├── 3.1 Baseline_Model.ipynb     # Baseline MLP model (same inputs as Black-Scholes)
│   ├── 3.2 Baseline_MoreParams.ipynb# Architecture tuning for baseline model
│   ├── 3.3 Extended_MLP.ipynb       # Extended MLP model with additional features
│   ├── 4. outlierAnalysis.ipynb     # Sensitivity outlier detection and local analysis
│   ├── Extra*.ipynb                 # Additional experiments with NeuralSens
│   └── utils.py                     # Utility functions
│
├── src/                # Core Python scripts
│   ├── optionScrapping.py           # Full scraping pipeline for option chains
│   └── dataCleansing.py             # Data cleaning and preprocessing pipeline
│
├── models/             # Saved model weights for reproducibility (.pkl files)
│   └── (Multiple saved MLP architectures for both baseline and extended models)
│
└── README.md           # This file
```

## Methodology Summary

The project follows the strucutre:

1. **Data Collection**:

- Real-world European options data from the S&P 500 index were collected using Yahoo Finance via the yfinance Python library.
- The scraping pipeline is implemented in src/optionScrapping.py and executed via 0. OptionScrapping.ipynb.

2. **Preprocessing**:

- Feature engineering steps include calculation of time-to-maturity, moneyness, mid-price, and filtering of low-quality contracts.
- Handled in 1. dataCleansing.ipynb and src/dataCleansing.py.

3. **Model Development**:

- Two models are developed:
  - Baseline MLP: replicates Black-Scholes inputs (S, K, T, σ).
  - Extended MLP: adds open interest, volume and moneyness features.
  - Training, hyperparameter tuning and evaluation conducted in 3.\* notebooks.

4. **Performance Comparison**:

- Classical metrics used: RMSE, MAE, R².
- Pairwise comparison against Black-Scholes performance on real data.

5. **Interpretability Analysis**:

- NeuralSens library is used to compute partial derivatives (Jacobian-based sensitivities).
- Global analysis: distribution of sensitivities and $\alpha$-curves.
- Local analysis: sensitivity outliers vs Black-Scholes Greeks.

6. **Synthetic Experiments**:

- Additional controlled experiments using synthetic option datasets located in test/ folder.
- Code for synthetic tests in Extra\*.ipynb.

## Dependencies

The code was developed using:

- Python 3.10
- yfinance
- BeautifulSoup
- scikit-learn
- numpy, pandas, matplotlib, seaborn
- NeuralSens (interpretability)
- joblib (for model persistence)

Exact dependencies and versions can be exported via:

```
pip freeze > requirements.txt
```

## Reproducibility

All data files are included to allow full reproducibility. The exact execution order is:

0. OptionScrapping.ipynb
1. dataCleansing.ipynb
2. EDA.ipynb
3. Model trainings
4. outlierAnalysis.ipynb
5. Extra\* notebooks for interpretability experiments.

## Author

Juan Sánchez Fernández

Master Thesis - Universidad Pontificia Comillas (ICAI)

Academic Year 2024/2025
