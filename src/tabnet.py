from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import root_mean_squared_error

class TabNetTrainer:
    def __init__(self, params=None):
        default_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'lambda_sparse': 1e-4,
            'optimizer_fn': 'adam',
            'optimizer_params': dict(lr=2e-2)
        }
        self.params = params if params else default_params
        self.model = TabNetRegressor(**self.params)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size = 0, patience=10):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['rmse'],
            max_epochs=epochs,
            patience=patience
        )
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"TabNet RMSE: {rmse}")
        return rmse
