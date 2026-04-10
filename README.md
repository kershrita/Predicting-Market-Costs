# Market Price Prediction System

Machine learning regression system for predicting product costs in noisy, real-world market data.

## Overview

This project was built as an end-to-end applied AI system, not only as a model training notebook.
It addresses a core business problem: estimating product prices from structured retail-market signals with varying data quality and inconsistent input schemas.

Project window:
- Aug 2023 to Sep 2023

System focus:
- Data ingestion and schema-aware preprocessing
- Feature engineering for market behavior and product context
- Regression model selection, tuning, and evaluation
- Reproducible experimentation across multiple algorithms

Real-world use cases:
- Pricing intelligence for retail planning teams
- Pre-quote cost estimation for procurement workflows
- Margin-risk monitoring for campaigns and promotions

## Competition Context

- Built for the IEEE MANSB Victoris 2 Kaggle competition:
	- https://www.kaggle.com/competitions/ieee-mansb-victoris-2/
- Team name used in competition submissions:
	- Champion Team

## Original Kaggle Notebooks

- Data Pre-processing:
	- https://www.kaggle.com/code/kershrita/market-costs-data-preprocessing-guide
- Model Building:
	- https://www.kaggle.com/code/kershrita/market-costs-model-feature-engineering-guide

## Data Description

This section preserves the original dataset field definitions from the previous README.

| Column Name | Description |
|---|---|
| Person Description | Description of the person visiting the market |
| Place Code | Code for each place which consists of 2 city code parts separated by "_" |
| Customer Order | Order of each customer in the market |
| Additional Features in market | A list of features that are found in the market |
| Promotion Name | Name of promotion made by the market on media |
| Store Kind | Genre or category of the store |
| Store Cost | Cost of the store |
| Store Sales | Amount of money spent on sales since the store first opened |
| Gross Weight | Weight of the bought item |
| Net Weight | Weight of bought item without packaging |
| Package Weight | Weight of the packaging |
| Is Recyclable? | Whether the item is recyclable or not |
| Yearly Income | Minimum income for the consumer per year |
| Store Area | Area of the store |
| Grocery Area | Area of the grocery department in the store |
| Frozen Area | Area of the frozen food department in the store |
| Meat Area | Area of the meat department in the store |
| Cost | Target variable to be predicted |

## Architecture

The system follows an end-to-end regression pipeline:

![Market Price Prediction System Pipeline](./assets/Market%20Price%20Prediction%20System%20Pipeline.png)

Core components:
- Input layer
	- Raw tabular market data from train and test CSV files
	- Handles naming inconsistencies across datasets
- Preprocessing layer
	- Parsing composite text fields into structured columns
	- Type normalization, unit scaling, and robust missing-value treatment
- Feature engineering layer
	- Domain-driven engineered predictors (promotion, store efficiency, income segmentation, product metadata)
- Modeling layer
	- Broad benchmark of regression algorithms
	- Dedicated tuned models (including CatBoost and tree-based ensembles)
- Evaluation and output layer
	- RMSE, MAE, and R2-centered evaluation strategy
	- Candidate model comparison and result generation

## Features

- End-to-end tabular regression workflow from raw data to final predictions
- Schema-resilient preprocessing for multiple column-name variants
- Domain-specific feature engineering for customer, promotion, and store attributes
- Robust null handling with iterative imputation for numeric features
- Model benchmarking across a wide set of regressors
- Hyperparameter search with GridSearchCV for selected models
- Notebook-driven experimentation with reusable preprocessing functions

## Technical Highlights

- Parsing composite, semi-structured fields into model-ready columns
	- Person Description -> marital status, gender, children, education, work
	- Place Code -> store and country identifiers
	- Customer Order -> order and department attributes
- Multi-label market amenities encoded into binary signals plus aggregated Amenities Score
- Defensive schema handling for real-world data drift
	- Supports alternate weight and income field names across files
- Engineered business features that improve signal quality
	- Family Expenses
	- Store Efficiency
	- Promotion Name Length and Promotion Frequency
	- Income Level and Price Tier
	- Order Popularity
- Structured evaluation track
	- Broad benchmark from linear to gradient/tree ensembles
	- Tuned candidate experiments logged in artifacts

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- XGBoost
- LightGBM
- Jupyter Notebook

## Getting Started

### 1) Clone Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd Predicting-Market-Costs
```

### 2) Create Environment and Install Dependencies

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install pandas numpy scikit-learn catboost xgboost lightgbm matplotlib seaborn jupyter
```

### 3) Run the Pipeline Notebooks

Execute in order:
- 1 - Data Preprocessing.ipynb
- 2 - EDA.ipynb
- 3 - Model Building.ipynb

### 4) Reuse Preprocessing as Module

The preprocessing and wrangling utilities are available in:
- pre_processing_funcs.py

## Results

The project includes two complementary experiment tracks: a broad model benchmark and dedicated CatBoost training artifacts.

Benchmark highlights (from model comparison output):
- ExtraTreesRegressor
	- RMSE: 44.07
	- R2: 0.92
- RandomForestRegressor
	- RMSE: 48.47
	- R2: 0.90
- XGBRegressor
	- RMSE: 51.61
	- R2: 0.89
- BaggingRegressor
	- RMSE: 50.99
	- R2: 0.89

CatBoost artifact highlights:
- Test RMSE improved from 148.47 at iteration 0 to 53.8798 at iteration 258
- Learn RMSE reached 55.6322 at iteration 258

Evaluation metrics used throughout the project:
- RMSE
- MAE
- R2

## Model Details

- Problem type: supervised regression
- Input type: structured tabular market data
- Target: Cost
- Training strategy:
	- train_test_split-based validation
	- algorithm benchmark for baseline comparison
	- focused tuning for ensemble models
	