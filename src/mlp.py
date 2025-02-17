import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[128, 64], dropout_rate=0.2):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units[1], 1)  # Regression output
        )
    
    def forward(self, x):
        return self.model(x)

class MLPTrainer:
    def __init__(self, input_dim, hidden_units=[128, 64], dropout_rate=0.2, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPModel(input_dim, hidden_units, dropout_rate).to(self.device)
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        self.model.train()
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)

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
        print(f"MLP RMSE: {rmse:.4f}")
        return rmse
