
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def clean_ccs_data(file_path):
# Loads and cleans the SOCAT Dataset
    df = pd.read_csv(file_path, skiprows=32)
# select and rename the columns
    df = df[[
        'WOA_SSS', 'TEMP', 'LATITUDE', 'LONGITUDE',
        'PRESSURE_ATM', 'PRESSURE_EQUI',
        'DIST_TO_LAND', 'ETOPO2',
        'GVCO2',
        'YEAR', 'MONTH',
        'FCO2_RECOMMENDED'
]]
    df.columns = [
        'Salinity', 'Temperature', 'Latitude', 'Longitude',
        'Pressure_atm', 'Pressure_equi',
        'Dist_to_land', 'Depth',
        'GVCO2',
        'Year', 'Month',
        'Target'
    ]
    # Handle missing/placeholder values
    df = df.replace([-1e34, -999], np.nan)
    df = df.dropna()
    # Remove duplicates
    df = df.drop_duplicates()
    #Filter by physical constraints
    df = df[
        (df['Salinity'] > 0) & (df['Salinity'] <= 40) &
        (df['Temperature'] > -2) & (df['Temperature'] < 40) &
        (df['Target'] > 0) & (df['Target'] < 2000)
    ]
    
    return df
def split_data_by_year(df, split_year=2020):
# Sorts data and performs a temporal split based on a specific year.
    # Sort to ensure chronological order
    df = df.sort_values('Year')
    # Temporal split
    train = df[df['Year'] < split_year]
    test = df[df['Year'] >= split_year]

    # Separate Features and Target
    X_train = train.drop(columns=['Target'])
    y_train = train['Target']

    X_test = test.drop(columns=['Target'])
    y_test = test['Target']

    return X_train, X_test, y_train, y_test

def split_data(df, test_size=0.2, random_state=42):
    
    #Separates the target and splits the data into training and testing sets.
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test