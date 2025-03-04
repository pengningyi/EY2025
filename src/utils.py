import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import rasterio
from pyproj import Proj, Transformer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data):
    """Load data from a CSV file."""
    X = data.drop(columns=["UHI Index", "Longitude", "Latitude"])
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
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=False)
    val_size_adjusted = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=seed, shuffle=False)
    
    if data_type == 'tensor':
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
    return X_train, X_val, X_test, y_train, y_val, y_test



def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def combine_two_datasets(dataset1,dataset2):    
    #Returns a  vertically concatenated dataset
    data = pd.concat([dataset1,dataset2], axis=1)
    return data


def get_transform_crs(tiff_path):
        # Get the transform and CRS from one of the original GeoTIFF files
    with rasterio.open(tiff_path) as src:
        transform = src.transform
        crs = src.crs
    return transform, crs


def remove_dup(df, cols):
    for col in cols:
        # Check if the value is a numpy array and has more than one dimension
        df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

    uhi_data = df.drop_duplicates(subset=cols, keep='first')
    uhi_data=uhi_data.reset_index(drop=True)

    return uhi_data


def remove_features(df, cols):
    df = df.drop(cols, axis=1)
    return df


def remove_missing_values(df):
    df = df.dropna()
    return df


def feature_selection(data, target):
    """
    Select relevant features from the mapped data.
    """
    # Assuming the target variable is in the CSV file and named 'Target'
    features = data.drop(target, axis=1)
    target = data[target]
    
    return features, target


def scale_features(train, test):
    sc = StandardScaler()
    train_scaled = sc.fit_transform(train)
    test_scaled = sc.transform(test)
    return sc, train_scaled, test_scaled


def model_training_rf(train_features, train_target):
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_features, train_target)
    return model

def prediction(model, features):
    # Predict the target variable for the training data
    predictions = model.predict(features)
    return predictions


def plot_feature_importance(model, features):
    # Plot the feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=features.columns[indices], palette='viridis')
    plt.title("Feature Importance")
    plt.show()



def calculate_cross_val_score(model, features, target):
    # Calculate the cross-validated RMSE
    cv_rmse = np.sqrt(-calculate_cross_val_score(model, features, target, scoring="neg_mean_squared_error", cv=5))
    return cv_rmse


def r2_score(target, predictions):
    # Calculate the R^2 score
    r2 = r2_score(target, predictions)
    return r2



def generate_submission(trainer, test_data_path = '../data/Submission_template.csv', data_type = 'tensor'):
    """
    Generate predictions and create a submission file.
    """
    test_data = pd.read_csv(test_data_path)

    from data import get_sentinel_data
    val_data = get_sentinel_data(dataset="test")

    submission_val_data=val_data.drop(columns = ["Latitude", "Longitude"])

    submission_val_data = submission_val_data.values
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    transformed_submission_data = scaler.fit_transform(submission_val_data)
    
    if data_type == 'tensor':
        transformed_submission_data = torch.tensor(transformed_submission_data, dtype=torch.float32)

    final_predictions = trainer.predict(transformed_submission_data)
    final_prediction_series = pd.Series(final_predictions)

    submission_df = pd.DataFrame({'Longitude':test_data['Longitude'].values, 'Latitude':test_data['Latitude'].values, 'UHI Index':final_prediction_series.values})

    submission_df.to_csv("../submission/submission.csv",index = False)
