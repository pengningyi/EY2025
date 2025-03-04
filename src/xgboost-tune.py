import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np
from utils import load_data, scale_data, split_data
from data import get_sentinel_data

print("Loading data...")
X, y = load_data(get_sentinel_data())
X = scale_data(X)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.2, 0.1, 42, data_type="dataframe")


def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 10.0),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "random_state": 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred = model.predict(X_val)
    mae = root_mean_squared_error(y_val, y_pred)
    
    return mae


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

print("Best params:", study.best_params)
