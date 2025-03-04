import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from loss import huber_loss, custom_weighted_rmse

class WideAndDeepModel(nn.Module):
    def __init__(self, input_dim, wide_units=128, deep_units=[128, 64]):
        super(WideAndDeepModel, self).__init__()
        # Wide part
        self.wide = nn.Linear(input_dim, wide_units)
        
        # Deep part
        deep_layers = []
        for i in range(len(deep_units)):
            if i == 0:
                deep_layers.append(nn.Linear(input_dim, deep_units[i]))
            else:
                deep_layers.append(nn.Linear(deep_units[i-1], deep_units[i]))
            deep_layers.append(nn.ReLU())
        self.deep = nn.Sequential(*deep_layers)
        
        # Combined output
        self.output = nn.Linear(wide_units + deep_units[-1], 1)
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.output(combined)

class WideAndDeepTrainer:
    def __init__(self, input_dim, wide_units=128, deep_units=[128, 64], learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WideAndDeepModel(input_dim, wide_units, deep_units).to(self.device)
        self.criterion = nn.HuberLoss()  # Mean Squared Error for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        self.model.train()
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(self.device), torch.tensor(y_val, dtype=torch.float32).to(self.device)

        for epoch in range(epochs):
            # Training
            self.model.train()
            permutation = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = self.criterion(val_outputs, y_val).item()
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
    
    def predict(self, X_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_test).squeeze().cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        huber = huber_loss(y_test, y_pred)
        custom_rmse = custom_weighted_rmse(y_test, y_pred)
        print(f"RMSE: {rmse}, MAE: {mae}, Huber: {huber}, Custom RMSE: {custom_rmse}")
        print(f"XGBoost RMSE: {rmse}")
        return huber