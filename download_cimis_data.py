"""
Script to download CIMIS station data for time period 2004-2023
Based on get_CIMIS.ipynb by John Franco Saraceno

This script:
- Filters stations that were active during 2004-2023
- Alternates API keys after each station download
- Downloads data in yearly chunks to avoid API limits
- Saves each station as a separate CSV file
"""

import datetime
import dateutil
import json
import pandas as pd
import urllib.request as urllibr
import urllib.error as urllibe
import time
import os
from dateutil.relativedelta import relativedelta

# API Keys
appKey = '3bbd5b81-3484-48fc-8de9-cf586df687ed'
appKey2 = '738e7081-2b8c-4f22-8d62-61dc5243683d'

# Configuration
START_DATE = '2004-01-01'
END_DATE = '2023-12-31'
OUTPUT_DIR = '/group/moniergrp/SpatialCIMIS/CIMIS/test/'
INTERVAL = 'spatial_more'  # Options: 'daily', 'hourly', 'default', 'spatial_more'


def retrieve_cimis_station_info(verbose=False):
    """Retrieve CIMIS station information from API"""
    StationNbr = []
    Name = []
    station_url = 'http://et.water.ca.gov/api/station'
    try:
        response = urllibr.urlopen(station_url)
        response_bytes = response.read()
        response_str = response_bytes.decode('utf-8')
        content = json.loads(response_str)
        stations = content['Stations']
        for i in stations:
            if i['IsActive'] == "True":
                StationNbr.append(i['StationNbr'])
                Name.append(i['Name'])
        if verbose is True:
            return stations
        else:
            return dict(zip(StationNbr, Name))
    except urllibe.HTTPError as e:
        print(f"There was an HTTPError when querying CIMIS for station information: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error when querying CIMIS: {e}")
        return None


def filter_stations_by_date_range(stations_list, connect_date_filter='2004-01-01', 
                                   disconnect_date_filter='2023-12-31'):
    """
    Filter stations that were active during the specified date range.
    
    Args:
        stations_list: List of station dictionaries from API
        connect_date_filter: Start date to filter by (YYYY-MM-DD)
        disconnect_date_filter: End date to filter by (YYYY-MM-DD)
    
    Returns:
        List of station numbers that were active during the period
    """
    connect_filter = datetime.datetime.strptime(connect_date_filter, '%Y-%m-%d')
    disconnect_filter = datetime.datetime.strptime(disconnect_date_filter, '%Y-%m-%d')
    
    active_stations = []
    
    for station in stations_list:
        # Parse dates
        try:
            connect_date = datetime.datetime.strptime(station.get('ConnectDate', '1970-01-01'), '%m/%d/%Y')
            disconnect_date = datetime.datetime.strptime(station.get('DisconnectDate', '2050-12-31'), '%m/%d/%Y')
        except (ValueError, KeyError):
            continue
        
        # Check if station was active during our period
        if connect_date <= disconnect_filter and disconnect_date >= connect_filter:
            active_stations.append(str(station['StationNbr']))
    
    return active_stations


def retrieve_cimis_data(url, target, stations_dict=None):
    """Retrieve CIMIS data from API for a given URL and station"""
    try:
        if stations_dict is None:
            stations_dict = retrieve_cimis_station_info()
        
        station_name = stations_dict.get(str(target), f"Station {target}")
        response = urllibr.urlopen(url)
        response_bytes = response.read()
        response_str = response_bytes.decode('utf-8')
        print(f'Retrieving data for {station_name} (Station #{target})')
        return json.loads(response_str)
    except urllibe.HTTPError as e:
        station_name = stations_dict.get(str(target), f"Station {target}") if stations_dict else f"Station {target}"
        print(f"Could not resolve the http request for {station_name}")
        error_msg = e.read().decode('utf-8') if hasattr(e.read(), 'decode') else str(e.read())
        print(error_msg)
        if e.code == 400 and 'The report request exceeds the maximum data limit' in error_msg:
            print("Shorten the requested period of record. Try limiting the number of parameters or a maximum of 30 days for hourly data.")
        return None
    except urllibe.URLError as e:
        print(f'Could not access the CIMIS database. Verify that you have an active internet connection and try again.')
        print(str(e))
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error when retrieving data: {e}")
        return None


def convert_data_items(ItemInterval):
    """Convert interval to data items list"""
    if ItemInterval == 'daily':
        dataItems_list = ['day-air-tmp-avg',
                          'day-air-tmp-max',
                          'day-air-tmp-min',
                          'day-sol-rad-avg',
                          'day-eto',
                          'day-asce-eto']
    elif ItemInterval == 'spatial':
        dataItems_list = ['day-sol-rad-avg', 'day-asce-eto']
    elif ItemInterval == 'spatial_more':
        dataItems_list = ['day-air-tmp-max',
                          'day-air-tmp-min',
                          'day-rel-hum-avg',
                          'day-dew-pnt',
                          'day-wind-spd-avg']
    elif ItemInterval == 'hourly':
        dataItems_list = ['hly-air-tmp',
                          'hly-dew-pnt',
                          'hly-eto',
                          'hly-net-rad',
                          'hly-asce-eto',
                          'hly-asce-etr',
                          'hly-precip',
                          'hly-rel-hum',
                          'hly-res-wind',
                          'hly-soil-tmp',
                          'hly-sol-rad',
                          'hly-vap-pres',
                          'hly-wind-dir',
                          'hly-wind-spd']
    elif ItemInterval == 'default':
        dataItems_list = ['day-asce-eto',
                          'day-precip',
                          'day-sol-rad-avg',
                          'day-vap-pres-avg',
                          'day-air-tmp-max',
                          'day-air-tmp-min',
                          'day-air-tmp-avg',
                          'day-rel-hum-max',
                          'day-rel-hum-min',
                          'day-rel-hum-avg',
                          'day-dew-pnt',
                          'day-wind-spd-avg',
                          'day-wind-run',
                          'day-soil-tmp-avg']
    else:
        dataItems_list = ['day-air-tmp-avg']
    
    dataItems = ','.join(dataItems_list)
    return dataItems


def cimis_to_dataframe(appKey, station, start, end, dataItems, stations_dict=None):
    """Fetch data from CIMIS API and return as DataFrame"""
    url = ('http://et.water.ca.gov/api/data?appKey=' + appKey + '&targets='
            + str(station) + '&startDate=' + start + '&endDate=' + end +
            '&dataItems=' + dataItems +'&unitOfMeasure=M')
    print(f'URL: {url}')

    data = retrieve_cimis_data(url, station, stations_dict)
    
    if data is None:
        return None
    
    try:
        dataframe = pd.json_normalize(data, record_path=['Data', 'Providers', 'Records'])
        return dataframe
    except (KeyError, TypeError) as e:
        print(f"Error parsing data for station {station}: {e}")
        return None


def download_station_data(appKey, station, start_date_str, end_date_str, interval, stations_dict):
    """Download data for a single station, breaking into yearly chunks if needed"""
    dataItems = convert_data_items(interval)
    startDate = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    endDate = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Get the years in the range
    years = range(startDate.year, endDate.year + 1)
    
    print(f'Downloading data for {len(years)} years: {list(years)[0]}-{list(years)[-1]}')
    
    # Create year-by-year date ranges
    if len(years) > 1:
        tstart = [str(datetime.date(yr, 1, 1)) for yr in years]
        tend = [str(datetime.date(yr, 12, 31)) for yr in years]
        tstart[0] = str(startDate.date())
        tend[-1] = str(endDate.date())
    else:
        tstart = [str(startDate.date())]
        tend = [str(endDate.date())]
    
    print(f'Date ranges: {list(zip(tstart, tend))}')
    
    # Collect all data
    all_dataframes = []
    for sd, ed in zip(tstart, tend):
        print(f'  Downloading {sd} to {ed}...')
        dataframe = cimis_to_dataframe(appKey, station, sd, ed, dataItems, stations_dict)
        
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            all_dataframes.append(dataframe)
            print(f'  Successfully downloaded {len(dataframe)} records')
        
        # Small delay between requests to be respectful
        time.sleep(1)
    
    if all_dataframes:
        # Combine all yearly dataframes
        site_data = pd.concat(all_dataframes, ignore_index=True)
        return site_data
    else:
        return None


def main():
    """Main function to download CIMIS data"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all station info
    print("Fetching station information from CIMIS API...")
    all_stations = retrieve_cimis_station_info(verbose=True)
    
    if all_stations is None:
        print("Failed to retrieve station information. Exiting.")
        return
    
    # Build station names dictionary from all stations (not just currently active)
    # This ensures we have names for stations that were active in 2004-2023 but are now inactive
    stations_dict = {}
    for station in all_stations:
        station_num = str(station.get('StationNbr', ''))
        station_name = station.get('Name', f'Station_{station_num}')
        if station_num:
            stations_dict[station_num] = station_name
    
    # Filter stations active during 2004-2023
    print("\nFiltering stations active during 2004-2023...")
    active_station_nums = filter_stations_by_date_range(all_stations, 
                                                         connect_date_filter='2004-01-01',
                                                         disconnect_date_filter='2023-12-31')
    
    print(f"Found {len(active_station_nums)} stations active during 2004-2023")
    
    # Alternate between API keys
    api_keys = [appKey, appKey2]
    key_index = 0
    
    # Download data for each station
    for idx, station_num in enumerate(active_station_nums):
        # Alternate API key
        current_key = api_keys[key_index % len(api_keys)]
        key_index += 1
        
        print(f"\n{'='*60}")
        print(f"Processing station {idx + 1}/{len(active_station_nums)}: #{station_num}")
        print(f"Using API key: {'Key 1' if current_key == appKey else 'Key 2'}")
        print(f"{'='*60}")
        
        try:
            # Download data
            site_data = download_station_data(current_key, station_num, 
                                             START_DATE, END_DATE, 
                                             INTERVAL, stations_dict)
            
            if site_data is not None and not site_data.empty:
                # Get station name for filename
                station_name = stations_dict.get(station_num, f"Station_{station_num}")
                # Clean filename
                safe_name = station_name.replace("/", "_").replace(" ", "_")
                csv_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{station_num}.csv")
                
                # Save to CSV
                site_data.to_csv(csv_path, index=False)
                print(f"✓ Saved {len(site_data)} records to {csv_path}")
            else:
                print(f"✗ No data retrieved for station {station_num}")
        
        except Exception as e:
            print(f"✗ Error processing station {station_num}: {e}")
        
        # Wait between stations to avoid rate limiting
        if idx < len(active_station_nums) - 1:
            wait_time = 60
            print(f"\nWaiting {wait_time} seconds before next station...")
            time.sleep(wait_time)
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Data saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

