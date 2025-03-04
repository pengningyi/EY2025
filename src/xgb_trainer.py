from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

class XGBoostTrainer:
    def __init__(self, learning_rate, n_estimators, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda, random_state):
        params = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,  # L1 regularization (default is higher, reduce it)
            'reg_lambda': reg_lambda,  # L2 regularization (same)
            'random_state': random_state
        }
        self.model = XGBRegressor(**params)

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"XGBoost RMSE: {rmse}")
        return rmse
