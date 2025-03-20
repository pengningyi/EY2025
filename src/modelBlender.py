from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mlp import MLPTrainer
from cnn import CNN1DTrainer
from sklearn.metrics import mean_squared_error
import numpy as np
import torch

class ModelBlender:
    def __init__(self, input_dimension):
        # Default hyperparameters
        self.xgb_params = {
            # 'n_estimators': 1000, 'learning_rate': 0.005, 'max_depth': 12,
            # 'subsample': 0.6, 'colsample_bytree': 0.7, 'random_state': 42
            
        "epochs": 20,
        "batch_size": 32,
        "lr": 0.001,
        "seed": 42,
        "dropout_rate": 0.3,
        "max_depth": 10,
        "n_estimators": 500
        }
        
        # self.lgb_params = lgb_params if lgb_params else {
        #     'n_estimators': 1000, 'learning_rate': 0.005, 'max_depth': 6,
        #     'subsample': 0.6, 'colsample_bytree': 0.7, 'random_state': 42
        # }
        
        # self.cat_params = cat_params if cat_params else {
        #     'iterations': 1200,
        #     'learning_rate': 0.01,
        #     'depth': 10,
        #     'l2_leaf_reg': 5,
        #     'random_seed': 42,
        #     'verbose': 0
        # }
        
        self.mlp_params = {
            "learning_rate": 0.005,
            "dropout_rate": 0.3
        }
        
        self.cnn1d_params = {
            "learning_rate": 0.0001,
            "hidden_units": [
                128,
                256
            ],
            "dropout_rate": 0.5
        }

        # Initialize models
        self.xgb = XGBRegressor(**self.xgb_params)
        self.cnn1d = CNN1DTrainer(input_dim=input_dimension, **self.cnn1d_params)
        self.mlp = MLPTrainer(input_dim=input_dimension, **self.mlp_params)
        
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        print("Training XGBoost...")
        self.xgb.fit(X_train, y_train)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        print("Training CNN1D...")
        self.cnn1d.fit(X_train, y_train, X_val, y_val)
        print("Training MLP...")
        self.mlp.fit(X_train, y_train, X_val, y_val)
    def predict(self, X_test, weights=(0.4, 0.3, 0.3)):
        pred_xgb = self.xgb.predict(X_test)
        pred_cnn1d = self.cnn1d.predict(X_test)
        pred_mlp = self.mlp.predict(X_test)
        
        if not isinstance(pred_xgb, np.ndarray):
            pred_xgb = np.array(pred_xgb)
        if not isinstance(pred_cnn1d, np.ndarray):
            pred_cnn1d = np.array(pred_cnn1d)
        if not isinstance(pred_mlp, np.ndarray):
            pred_mlp = np.array(pred_mlp)

        # 打印形状以调试
        print(f"pred_xgb shape: {pred_xgb.shape}")
        print(f"pred_cnn1d shape: {pred_cnn1d.shape}")
        print(f"pred_mlp shape: {pred_mlp.shape}")


        final_pred = (weights[0] * pred_xgb + weights[1] * pred_cnn1d + weights[2] * pred_mlp)
        return final_pred
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Blended Model RMSE: {rmse}")
        return rmse
