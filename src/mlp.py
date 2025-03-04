import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[128, 64], dropout_rate=0.3):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class MLPTrainer:
    def __init__(self, input_dim, hidden_units=[128, 64], dropout_rate=0.3, learning_rate=0.001): #dropout_rate from 0.2 to 0.3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPModel(input_dim, hidden_units, dropout_rate).to(self.device)
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5) #from Adam to AdamW
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, early_stop_patience=10):
        self.model.train()
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)

        best_val_loss = float('inf')
        best_model_state = None
        patience = 0
        
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1

            if patience >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
    
    def predict(self, X_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_test).squeeze()
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"MLP RMSE: {rmse:.4f}")
        return rmse
