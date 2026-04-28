import pandas as pd
import joblib
# Handle model saving, loading, and predictions for CO2 solubility. 

def save_model(model, path):
# Save trained model to disk.
    joblib.dump(model, path)

def load_model(path):
#Load trained model from disk.
    return joblib.load(path)

def predict_co2(model, input_data):
#Predict CO2 values for multiple samples.
    return model.predict(input_data)

def predict_single(model, input_dict):
    # Predict CO2 for a single sample.
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]