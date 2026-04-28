# CO₂ Solubility Prediction in Brine Systems using ML Models

## Overview
This project predicts CO₂ solubility in ocean water using machine learning models and integrates carbon chemistry computations to derive ion concentrations (HCO₃⁻, CO₃²⁻, CO₂). It combines environmental data with ML techniques to support carbon cycle analysis and climate research.

---

## Objectives
- Predict CO₂ solubility using oceanographic features
- Compare multiple ML models for best performance
- Integrate carbon chemistry for ion estimation
- Analyze model performance using statistical metrics

---

## Dataset
This project uses the SOCAT(Surface Ocean CO₂ Atlas) dataset.
The dataset includes ocean parameters such as:
- Sea Surface Temperature
- Salinity
- Atmospheric Pressure
- Latitude & Longitude
- Depth
- Distance to land
- Year abd Month

Target variable:
- CO₂ concentration (FCO2_RECOMMENDED)

---

## Machine Learning Models Used
- Random Forest Regressor
- XGBoost
- LightGBM
- CatBoost
- K-Nearest Neighbors
- MLP Regressor

---

##  Methodology

### 1. Data Preprocessing
- Missing value handling
- Removal of duplicates
- Filtering based on physical constraints
- Train-test split

### 2. Model Training
- Multiple regression models trained
- Hyperparameters tuned for performance

### 3. Evaluation Metrics
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### 4. Chemistry Integration
- Used PyCO2SYS-based computations
- Derived:
  - HCO₃⁻
  - CO₃²⁻
  - CO₂

---

## Results
- Best model selected based on R² score
- Comparison of all models stored in `results/model_results.csv`
- Prediction outputs stored for analysis

---

##  Project Structure
```
co2-solubility-ml/
├── data/               # Raw and processed brine datasets
├── notebooks/          # Exploratory Data Analysis (EDA)
├── results/            # Model plots, CSV outputs, and residuals
├── src/                # Modular Python scripts
├── main.py             # Entry point to run the full pipeline
├── requirements.txt    # Library dependencies
└── README.md           # Project documentation
```
---

## Visualizations
- Model Comparison Plot
- Predicted vs Actual CO₂
- Residual Analysis
- Ion Concentration Comparison

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the project
```bash
python main.py
```


## Outputs

Generated files:
- model_results.csv
- predictions.csv
- ion_results.csv
- visualization plots (.png)