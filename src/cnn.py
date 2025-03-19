import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from loss import huber_loss, custom_weighted_rmse
import copy

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[64, 64], dropout_rate=0.3):
        super(CNN1DModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_units[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units[0]),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Conv1d(in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units[1]),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),

            nn.Flatten(),

            nn.Linear(hidden_units[1] * (input_dim // 4), 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class CNN1DTrainer:
    def __init__(self, input_dim, hidden_units=[64, 128], dropout_rate=0.3, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN1DModel(input_dim, hidden_units, dropout_rate).to(self.device)
        self.criterion = nn.HuberLoss()  # Huber Loss for regression
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)  # AdamW optimizer
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, early_stop_patience=10):
        self.model.train()
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

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

                # Reshape input for 1D-CNN
                batch_X = batch_X.unsqueeze(1)  # Add channel dimension

                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val.unsqueeze(1)).squeeze()
                val_loss = self.criterion(val_outputs, y_val).item()
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

            # Early stopping
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
        X_test = X_test.unsqueeze(1)  # Add channel dimension
        with torch.no_grad():
            return self.model(X_test).squeeze()
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred.cpu().numpy())
        mae = mean_absolute_error(y_test, y_pred.cpu().numpy())
        huber = huber_loss(y_test, y_pred.cpu().numpy())
        custom_rmse = custom_weighted_rmse(y_test, y_pred.cpu().numpy())
        print(f"RMSE: {rmse}, MAE: {mae}, Huber: {huber}, Custom RMSE: {custom_rmse}")
        return rmse