import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.chemistry import compute_ions


def evaluate_carbon_model(name, model, X_test, y_test, X_test_original):

    y_pred = model.predict(X_test)
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Safety checks
    if 'Salinity' not in X_test_original.columns or 'Temperature' not in X_test_original.columns:
        raise ValueError("Missing required columns for chemistry computation")

    # Chemistry 
    results = compute_ions(y_pred, X_test_original)

    return {
        "Model": name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "HCO3": results['HCO3'],
        "CO3": results['CO3'],
        "CO2": results['CO2']
    }