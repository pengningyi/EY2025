from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

class XGBoostTrainer:
    def __init__(self, params=None):
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.params = params if params else default_params
        self.model = XGBRegressor(**self.params)

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"XGBoost RMSE: {rmse}")
        return rmse
