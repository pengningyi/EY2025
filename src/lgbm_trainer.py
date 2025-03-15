import json
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from loss import huber_loss, custom_weighted_rmse
from utils import load_data, scale_data, split_data
from data import get_sentinel_data


class LGBMTrainer:
    def __init__(self, learning_rate=None, num_leaves=None, n_estimators=None, max_depth=None, subsample=None,
                 colsample_bytree=None, reg_alpha=None, reg_lambda=None, random_state=42, use_best_params=True):
        """
        Initializes the LGBMRegressor with either tuned parameters from `best_lgbm_params.json`
        or manually provided parameters.
        """
        
        # Load best parameters from JSON if available
        best_params_file = "best_lgbm_params.json"
        if use_best_params and os.path.exists(best_params_file):
            with open(best_params_file, "r") as f:
                best_params = json.load(f)
            print("Loaded best hyperparameters from", best_params_file)
        else:
            best_params = {
                "learning_rate": learning_rate if learning_rate else 0.01,
                "n_estimators": n_estimators if n_estimators else 1000,
                "max_depth": max_depth if max_depth else 10,
                "subsample": subsample if subsample else 0.8,
                "num_leaves": 31,
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "rmse",
                "random_state": random_state
            }
            print("Using manually specified parameters.")

        self.model = lgb.LGBMRegressor(**best_params)

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Trains the LightGBM model on the given dataset.
        """
        print("Training LGBMRegressor...")
        self.model.fit(X_train, y_train, 
                       eval_set=[(X_val, y_val)])

    def predict(self, X_test):
        """
        Makes predictions using the trained model.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model performance on test data.
        """
        y_pred = self.predict(X_test)
        
        rmse = mean_squared_error(y_test, y_pred)**0.5
        mae = mean_absolute_error(y_test, y_pred)
        huber = huber_loss(y_test, y_pred)
        custom_rmse = custom_weighted_rmse(y_test, y_pred)

        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Huber Loss: {huber:.4f}, Custom RMSE: {custom_rmse:.4f}")
        return rmse

"""
# Load dataset
print("Loading and preprocessing data...")
X, y = load_data(get_sentinel_data('C:\\Users\\T\\OneDrive\\Msc Macadamia\\EYChallenge\\EY2025\\data\\Training_data_uhi_index_2025-02-18.csv'))

X = scale_data(X)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.1, seed=42, data_type="dataframe")


# Try to load best parameters from tuning
best_params_file = "best_lgbm_params.json"
if os.path.exists(best_params_file):
    with open(best_params_file, "r") as f:
        best_params = json.load(f)
    print("Loaded best hyperparameters from", best_params_file)
else:
    best_params = {
        "learning_rate": 0.01,
        "n_estimators": 1000,
        "max_depth": 10,
        "num_leaves": 31,
        "subsample": 0.8,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42
    }
    print("Using default parameters.")

# Initialize LightGBM model
model = lgb.LGBMRegressor(**best_params)

# Train the model
print("Training LightGBM model...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Make predictions
print("Evaluating model...")
y_pred = model.predict(X_test)

# Compute evaluation metrics
#rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Take square root manually

mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
"
"""