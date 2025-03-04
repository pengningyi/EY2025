# Supress Warnings 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pyproj import Proj, Transformer
import rasterio
from tqdm import tqdm
import pystac_client
import planetary_computer 
from odc.stac import stac_load

lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)

bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

time_window = "2021-06-01/2021-09-01"

def save_index_as_tiff(index_data, filename, gt, width, height):
    """The calculated indices (NDVI, NDBI, NDWI) are derived from the spectral bands of the median composite. 
    These indices are used to highlight specific features or characteristics of the landscape, such as vegetation health (NDVI),
    built-up areas (NDBI), and water bodies (NDWI).
    Each index is calculated using specific bands from the median composite and is saved as a separate GeoTIFF file with a single band:"""
    
    with rasterio.open(filename, 'w', driver='GTiff', width=width, height=height,
                       crs='epsg:4326', transform=gt, count=1, compress='lzw', dtype='float64') as dst:
        dst.write(index_data.values, 1)
        
def map_satellite_data(combined_data, csv_path, transform, crs, buffer_radius=None):
    df = pd.read_csv(csv_path)
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values
    uhi = df['UHI Index'].values
    
    # print(df.columns)

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
    df['Latitude'] = latitudes
    df['Longitude'] = longitudes
    df['UHI Index'] = uhi
    return df


def get_sentinel_data(dataset = "train"):
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = stac.search(
        bbox=bounds, 
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 30}},
    )

    items = list(search.get_items())
    print('This is the number of scenes that touch our region:',len(items))
    
    signed_items = [planetary_computer.sign(item).to_dict() for item in items]

    # Define the pixel resolution for the final product
    # Define the scale according to our selected crs, so we will use degrees
    resolution = 10  # meters per pixel 
    scale = resolution / 111320.0 # degrees per pixel for crs=4326 
    
    BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    
    data = stac_load(
        items,
        bands=BANDS,
        crs="EPSG:4326", # Latitude-Longitude
        resolution=scale, # Degrees
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=planetary_computer.sign,
        bbox=bounds
    )
    
    median = data.median(dim="time").compute() #to reduce the impact of clouds.

    ndvi_median = (median.B08 - median.B04) / (median.B08 + median.B04)

    ndbi_median = (median.B11 - median.B08) / (median.B11 + median.B08)

    ndwi_median = (median.B03 - median.B08) / (median.B03 + median.B08)

    median_filename = "median_composite.tiff"
    ndvi_filename = "ndvi_median.tiff"
    ndbi_filename = "ndbi_median.tiff"
    ndwi_filename = "ndwi_median.tiff"

    height = median.dims["latitude"]
    width = median.dims["longitude"]
    gt = rasterio.transform.from_bounds(lower_left[1], lower_left[0], upper_right[1], upper_right[0], width, height)

    # Save the median composite
    #he median composite is saved as a GeoTIFF file with four bands
    with rasterio.open(median_filename, 'w', driver='GTiff', width=width, height=height,
                    crs='epsg:4326', transform=gt, count=12, compress='lzw', dtype='float64') as dst:
           
        dst.write(median.B01, 1)
        dst.write(median.B02, 2)
        dst.write(median.B03, 3)
        dst.write(median.B04, 4)
        dst.write(median.B05, 5)
        dst.write(median.B06, 6)
        dst.write(median.B07, 7)
        dst.write(median.B08, 8)
        dst.write(median.B8A, 9)
        dst.write(median.B11, 11)
        dst.write(median.B12, 12)
    
    save_index_as_tiff(ndvi_median, ndvi_filename, gt, width, height)
    save_index_as_tiff(ndbi_median, ndbi_filename, gt, width, height)
    save_index_as_tiff(ndwi_median, ndwi_filename, gt, width, height)

    # Load the median composite
    with rasterio.open("median_composite.tiff") as src:
        median_data = src.read()  # This will load all bands
    with rasterio.open("ndvi_median.tiff") as src:
        ndvi_data = src.read(1)  # Load the first (and only) band
    with rasterio.open("ndbi_median.tiff") as src:
        ndbi_data = src.read(1)  # Load the first (and only) band
    with rasterio.open("ndwi_median.tiff") as src:
        ndwi_data = src.read(1)  # Load the first (and only) band

    # Combine all data into a single array for the ML model
    # Assuming median_data has shape (bands, height, width)
    # and ndvi_data, ndbi_data, ndwi_data have shape (height, width)
    combined_data = np.vstack([
        median_data,
        ndvi_data[np.newaxis, :, :],  # Add a new axis to match the shape
        ndbi_data[np.newaxis, :, :],  # Add a new axis to match the shape
        ndwi_data[np.newaxis, :, :]   # Add a new axis to match the shape
    ])

    # Now combined_data has shape (bands + 3, height, width)
    # use this combined_data as input features for your ML model
    
    # Get the transform and CRS from one of the original GeoTIFF files
    with rasterio.open("median_composite.tiff") as src:
        transform = src.transform
        crs = src.crs

    col_name = BANDS + ["ndvi", "ndbi", "ndwi"]

    # Mapping satellite data with training data
    if dataset == "train":
        final_data = map_satellite_data(combined_data, '../data/Training_data_uhi_index_2025-02-18.csv', transform, crs, buffer_radius=50)
    else:
        final_data = map_satellite_data(combined_data, '../data/Submission_template.csv', transform, crs, buffer_radius=50)
        final_data.drop(columns = ["UHI Index"], inplace=True)
        
    return final_data