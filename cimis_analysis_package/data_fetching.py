# cimis_analysis_package/data_fetching.py

import json
import pandas as pd
import urllib.request
import geopandas as gpd
import os

# Import configuration constants
from . import config

def fetch_and_save_station_list_from_api(api_url=config.STATION_API_URL, 
                                         output_csv_path=config.DEFAULT_STATION_CSV_PATH):
    """
    Fetches station data from the CIMIS API and saves it to a CSV file.
    Args:
        api_url (str): The URL for the CIMIS station API.
        output_csv_path (str): Path to save the fetched station data.
    Returns:
        pd.DataFrame or None: DataFrame of station data if successful, else None.
    """
    try:
        print(f"Fetching station data from API: {api_url}")
        with urllib.request.urlopen(api_url) as response:
            content_bytes = response.read()
            content_str = content_bytes.decode('utf-8') # Ensure proper decoding
            content = json.loads(content_str)
            
        stations_data = content.get("Stations", None) # Use .get for safer access

        if stations_data:
            df = pd.DataFrame.from_dict(stations_data)
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            print(f"Successfully fetched and saved station data to {output_csv_path}")
            return df
        else:
            print("No 'Stations' key found in API response or data is empty.")
            return None
    except urllib.error.URLError as e:
        print(f"URL Error fetching station data: {e}. Check network or API URL.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error fetching station data: {e}. API response might not be valid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching or saving station data: {e}")
        return None

def load_and_prepare_station_list(csv_path=config.DEFAULT_STATION_CSV_PATH):
    """
    Loads the station list from a CSV file and prepares it as a GeoDataFrame.
    Args:
        csv_path (str): Path to the station list CSV file.
    Returns:
        gpd.GeoDataFrame or None: GeoDataFrame of prepared station data, else None.
    """
    try:
        if not os.path.exists(csv_path):
            print(f"Station CSV file not found at {csv_path}. Attempting to fetch from API.")
            fetch_and_save_station_list_from_api(output_csv_path=csv_path)
            if not os.path.exists(csv_path): # Check again if fetch failed
                 print(f"Failed to fetch station data. Cannot load station list from {csv_path}.")
                 return None

        print(f"Loading station list from: {csv_path}")
        station_list_df = pd.read_csv(csv_path)

        # Data cleaning and preparation
        station_list_df['HmsLatitude'] = station_list_df['HmsLatitude'].astype(str).str.split('/').str[-1].str.strip()
        station_list_df['HmsLongitude'] = station_list_df['HmsLongitude'].astype(str).str.split('/').str[-1].str.strip()
        station_list_df['Name'] = station_list_df['Name'].astype(str).str.replace(" ", "").str.replace("/", "").str.replace(".", "", regex=False).str.replace("-", "", regex=False)
        
        station_list_df['DisconnectDate'] = pd.to_datetime(station_list_df['DisconnectDate'], errors='coerce')
        station_list_df['ConnectDate'] = pd.to_datetime(station_list_df['ConnectDate'], errors='coerce')
        
        station_list_df.rename(columns={'HmsLatitude': 'Latitude', 'HmsLongitude': 'Longitude'}, inplace=True)
        
        station_list_df['Latitude'] = pd.to_numeric(station_list_df['Latitude'], errors='coerce')
        station_list_df['Longitude'] = pd.to_numeric(station_list_df['Longitude'], errors='coerce')
        
        # Drop rows where essential geo-coordinates are missing after coercion
        station_list_df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        if station_list_df.empty:
            print("No valid station data after cleaning coordinates.")
            return None

        stations_gdf = gpd.GeoDataFrame(
            station_list_df,
            geometry=gpd.points_from_xy(station_list_df.Longitude, station_list_df.Latitude),
            crs="EPSG:4326"  # Standard CRS for lat/lon
        )
        print(f"Station list loaded and prepared into GeoDataFrame. {len(stations_gdf)} stations.")
        return stations_gdf
    except FileNotFoundError: # Should be caught by os.path.exists now, but good fallback
        print(f"Error: Station CSV file not found at {csv_path} and could not be fetched.")
        return None
    except Exception as e:
        print(f"Error loading or preparing station list: {e}")
        return None

if __name__ == '__main__':
    # Example usage (and a simple test)
    print("Testing data_fetching module...")
    # 1. Try to load, which should trigger API fetch if file doesn't exist
    stations = load_and_prepare_station_list()
    if stations is not None:
        print("\nLoaded stations sample:")
        print(stations.head())
    else:
        print("\nFailed to load stations.")
