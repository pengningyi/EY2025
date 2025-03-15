import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from utils import load_data, scale_data, split_data
from data import get_sentinel_data

# Load and preprocess data
X, y = load_data(get_sentinel_data('C:\\Users\\T\\OneDrive\\Msc Macadamia\\EYChallenge\\EY2025\\data\\Training_data_uhi_index_2025-02-18.csv'))

X = scale_data(X)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.2, 0.1, 42, data_type="dataframe")

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred)**0.5
    
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)

# Save the best parameters
import json
with open("best_lgbm_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
