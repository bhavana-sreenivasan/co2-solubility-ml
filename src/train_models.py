from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_library():
    #Returns a dictionary of initialized ML models.
    
    models = {
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05),
        
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=5))
        ]),

        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),

        "CatBoost": CatBoostRegressor(
            verbose=0,
            depth=6,
            learning_rate=0.05,
            iterations=500,
            random_state=42
        ),

        "MLP": Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42
            ))
        ])
    }

    return models


def train_all_models(models, X_train, y_train):
    #Trains all models and returns trained versions.
    
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models