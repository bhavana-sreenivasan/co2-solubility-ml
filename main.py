import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from src.data_preprocessing import clean_ccs_data, split_data
from src.train_models import get_model_library, train_all_models
from src.evaluate import evaluate_all_models, results_to_dataframe
from src.visualize import (
    plot_model_comparison,
    plot_pred_vs_actual,
    plot_residuals,
    plot_ion_comparison
)


def main():
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)

    # Load & preprocess data
    df = clean_ccs_data("data/dataset.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    # Train models
    models = get_model_library()
    trained_models = train_all_models(models, X_train, y_train)

    # Evaluate models
    all_results = evaluate_all_models(
        trained_models,
        X_test,
        y_test,
        X_test_original=X_test.copy()
    )

    df_results = results_to_dataframe(all_results)
    df_results = df_results.sort_values("R2", ascending=False)

    # Save results
    df_results.to_csv("results/model_results.csv", index=False)
    print("\nModel Performance:\n", df_results)

    # Visualizations
    plot_model_comparison(df_results)
    plot_ion_comparison(df_results)

    # Best model
    best_model_name = df_results.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_result = next(res for res in all_results if res["Model"] == best_model_name)

    print(f"\nBest Model: {best_model_name}")

    # Save ion results
    ion_df = pd.DataFrame({
        "HCO3": best_result["HCO3"],
        "CO3": best_result["CO3"],
        "CO2": best_result["CO2"]
    })
    ion_df.to_csv("results/ion_results.csv", index=False)

    # Predictions
    y_pred = best_model.predict(X_test)
    pd.DataFrame({
    "Actual_CO2": y_test,
    "Predicted_CO2": y_pred
}).to_csv("results/predictions.csv", index=False)

    plot_pred_vs_actual(y_test, y_pred, best_model_name)
    plot_residuals(y_test, y_pred, best_model_name)


if __name__ == "__main__":
    main()