import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Model Comparison (R2, RMSE, MAE)
def plot_model_comparison(df_results):
    df = df_results.copy()

    # Normalize for fair comparison
    df['R2'] = df['R2'].clip(lower=0) / df['R2'].max()
    df['RMSE'] = df['RMSE'] / df['RMSE'].max()
    df['MAE'] = df['MAE'] / df['MAE'].max()

    df.plot(x='Model', y=['R2', 'RMSE', 'MAE'], kind='bar')

    plt.title("Model Performance Comparison (Normalized)")
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Predicted vs Actual
def plot_pred_vs_actual(y_test, y_pred, model_name="Model"):
    plt.scatter(y_test, y_pred, alpha = 0.6)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], color = 'red')

    plt.xlabel("Actual CO2")
    plt.ylabel("Predicted CO2")
    plt.title(f"{model_name}: Predicted vs Actual")
    plt.tight_layout()
    plt.show()


# Residual Plot
def plot_residuals(y_test, y_pred, model_name="Model"):
    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals)
    plt.axhline(y=0)

    plt.xlabel("Predicted CO2")
    plt.ylabel("Residuals")
    plt.title(f"{model_name}: Residual Plot")
    plt.tight_layout()
    plt.show()


#Ion Comparison (HCO3, CO3, CO2)
def plot_ion_comparison(df_results):
    df = df_results.copy()

    df.plot(x='Model', y=['HCO3_mean', 'CO3_mean', 'CO2_mean'], kind='bar')

    plt.title("Ion Concentration Comparison Across Models")
    plt.ylabel("Concentration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#CO₂ distribution
def plot_co2_distribution(all_results):
    data = []
    labels = []

    for res in all_results:
        data.append(res['CO2'])   # full CO2 values
        labels.append(res['Model'])

    plt.figure()
    plt.boxplot(data, labels=labels)

    plt.title("CO₂ Distribution Across Models")
    plt.ylabel("CO₂ Concentration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()