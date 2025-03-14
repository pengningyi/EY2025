from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

class ModelBlender:
    def __init__(self, xgb_params=None, lgb_params=None, cat_params=None):
        # Default hyperparameters
        self.xgb_params = xgb_params if xgb_params else {
            'n_estimators': 1000, 'learning_rate': 0.005, 'max_depth': 12,
            'subsample': 0.6, 'colsample_bytree': 0.7, 'random_state': 42
        }
        
        self.lgb_params = lgb_params if lgb_params else {
            'n_estimators': 1000, 'learning_rate': 0.005, 'max_depth': 6,
            'subsample': 0.6, 'colsample_bytree': 0.7, 'random_state': 42
        }
        
        self.cat_params = cat_params if cat_params else {
            'iterations': 1200,
            'learning_rate': 0.01,
            'depth': 10,
            'l2_leaf_reg': 5,
            'random_seed': 42,
            'verbose': 0
        }
        
        # Initialize models
        self.xgb = XGBRegressor(**self.xgb_params)
        self.lgb = LGBMRegressor(**self.lgb_params)
        self.cat = CatBoostRegressor(**self.cat_params)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        print("Training XGBoost...")
        self.xgb.fit(X_train, y_train)
        print("Training LightGBM...")
        self.lgb.fit(X_train, y_train)
        print("Training CatBoost...")
        self.cat.fit(X_train, y_train)
    
    def predict(self, X_test, weights=(0.5, 0.5, 0)):
        pred_xgb = self.xgb.predict(X_test)
        pred_lgb = self.lgb.predict(X_test)
        # pred_cat = self.cat.predict(X_test)
        pred_cat = 0

        final_pred = (weights[0] * pred_xgb + weights[1] * pred_lgb + weights[2] * pred_cat)
        return final_pred
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Blended Model RMSE: {rmse}")
        return rmse
