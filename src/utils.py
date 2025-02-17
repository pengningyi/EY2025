import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """Load data from a CSV file."""
    data = pd.read_csv(data_path)
    X = data.drop(columns=["UHI Index", "Longitude", "Latitude", "datetime"])
    y = data["UHI Index"]
    return X, y

def scale_data(X):
    """Scale data using StandardScaler."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def split_data(X, y, test_size=0.2, val_size=0.2, seed=42, data_type='dataframe'):
    """Split data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=seed)
    val_size_adjusted = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=seed)
    
    if data_type == 'tensor':
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
    return X_train, X_val, X_test, y_train, y_val, y_test
