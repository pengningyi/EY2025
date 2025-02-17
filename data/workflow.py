import rasterio
import pandas as pd
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

# Function to map satellite data to geo-locations
def map_satellite_data(combined_data, csv_path, transform, crs, buffer_radius=None):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_path)
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values

    # Convert lat/long to the GeoTIFF's CRS
    proj_wgs84 = Proj(init='epsg:4326')  # EPSG:4326 is the common lat/long CRS
    proj_tiff = Proj(crs)
    transformer = Transformer.from_proj(proj_wgs84, proj_tiff)

    band_values = {f'Band_{i+1}': [] for i in range(combined_data.shape[0])}

    # Iterate over the latitudes and longitudes, and extract the corresponding band values
    for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Mapping values"):
        x, y = transformer.transform(lat, lon)
        row, col = ~transform * (x, y)  # Transform to row, col in the array

        if buffer_radius:
            # Create a buffer around the location and calculate the average values within the buffer
            row_min = max(int(row - buffer_radius), 0)
            row_max = min(int(row + buffer_radius), combined_data.shape[1])
            col_min = max(int(col - buffer_radius), 0)
            col_max = min(int(col + buffer_radius), combined_data.shape[2])
            for i in range(combined_data.shape[0]):
                band_value = combined_data[i, row_min:row_max, col_min:col_max].mean()
                band_values[f'Band_{i+1}'].append(band_value)
        else:
            # Extract the nearest pixel value
            row = int(row)
            col = int(col)
            for i in range(combined_data.shape[0]):
                band_value = combined_data[i, row, col]
                band_values[f'Band_{i+1}'].append(band_value)

    # Create a DataFrame with the band values
    df = pd.DataFrame(band_values)

    return df


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


def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)  
    return X_train, X_test, y_train, y_test


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



def generate_submission(model, test_data_path, submission_path, tiff_path, csv_path):
    """
    Generate predictions and create a submission file.
    """
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    # Mapping satellite data for submission.
    val_data = map_satellite_data(tiff_path, combine_two_datasets)

    # Extracting specific columns (B01, B06, and NDVI) from the validation dataset
    submission_val_data=val_data.loc[:,[...]]
    
    # Feature Scaling 
    submission_val_data = submission_val_data.values
    
    transformed_submission_data = sc.transform(submission_val_data)
    
    #Making predictions
    final_predictions = model.predict(transformed_submission_data)
    final_prediction_series = pd.Series(final_predictions)
    
    #Combining the results into dataframe
    submission_df = pd.DataFrame({'Longitude':test_data['Longitude'].values, 'Latitude':test_data['Latitude'].values, 'UHI Index':final_prediction_series.values})

    #Dumping the predictions into a csv file.
    submission_df.to_csv("submission.csv",index = False)
