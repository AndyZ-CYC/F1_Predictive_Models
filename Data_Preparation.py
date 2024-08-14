import fastf1
import os
import pandas as pd
from fastf1 import utils

# Determine the path to the cache folder
script_directory = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
parent_directory = os.path.abspath(os.path.join(script_directory, '..'))  # Parent directory
cache_directory = os.path.join(parent_directory, 'f1_data_cache')  # Path to the cache folder

# Enable the cache for faster data retrieval
fastf1.Cache.enable_cache(cache_directory) 


def get_race_results(year, race_name):
    """
    Fetch the race results for a specific race and year.

    Parameters:
    year (int): The year of the race
    race_name (str): The name of the race (e.g., 'British Grand Prix')

    Returns:
    pd.DataFrame: A DataFrame containing the race results, including driver positions, lap times, and more.
                  The DataFrame contains various columns like Driver, Position, Team, etc.
    """
    session = fastf1.get_session(year, race_name, 'R')
    session.load(telemetry=False)  # Load session data
    results = session.results
    race_date = session.date  # Get the race date
    results['RaceDate'] = race_date  # Add the race date to the results DataFrame
    return results


def get_qualifying_results(year, race_name):
    """
    Fetch qualifying results for a specific race.

    Parameters:
    year (int): The year of the race
    race_name (str): The name of the race (e.g., 'British Grand Prix')

    Returns:
    pd.DataFrame: A DataFrame containing qualifying results, including qualifying positions and Q times
    """
    qualifying = fastf1.get_session(year, race_name, 'Q')
    qualifying.load(telemetry=False)  # Load session data
    qual_results = qualifying.results
    return qual_results[['DriverId', 'Position', 'Q1', 'Q2', 'Q3']]


def get_weather_data(year, race_name):
    """
    Fetch weather data for a specific race.

    Parameters:
    year (int): The year of the race
    race_name (str): The name of the race (e.g., 'British Grand Prix')

    Returns:
    pd.DataFrame: A DataFrame containing weather data for the specified race, 
                  including columns like 'AirTemp', 'TrackTemp', 'Humidity', 
                  'WindSpeed', 'WindDirection', and 'Rainfall'
    """
    session = fastf1.get_session(year, race_name, 'R')
    session.load(telemetry=False)  # Load session data
    weather_data = session.weather_data.drop(columns=['Time'])
    avg_weather = weather_data.mean()  # Taking average weather conditions
    return avg_weather.to_frame().T  # Return as DataFrame for consistency


def merge_race_and_qualifying(race_results, qual_results):
    """
    Merge race results with qualifying results.

    Parameters:
    race_results (pd.DataFrame): DataFrame containing race results
    qual_results (pd.DataFrame): DataFrame containing qualifying results

    Returns:
    pd.DataFrame: A DataFrame combining race and qualifying results
    """
    # Merge on common columns such as 'Driver'
    merged_results = pd.merge(race_results, qual_results, on='DriverId', how='left', suffixes=('_Race', '_Qual'))
    return merged_results


def prepare_f1_data(start_year, end_year, file_path):
    """
    This function retrieves race history data from the specified start year to end year, 
    merges it with qualifying results, and outputs a single dataset for each race.

    Parameters:
    start_year (int): The first year to include in the data
    end_year (int): The last year to include in the data

    Outputs:
    - 'f1_data_<start_year>_<end_year>.csv': CSV file containing merged race and qualifying data for all races within the specified year range
    """
    all_data = []
    for year in range(start_year, end_year + 1):
        schedule = fastf1.get_event_schedule(year)
        for race in schedule['EventName']:
            race_results = get_race_results(year, race)
            qual_results = get_qualifying_results(year, race)
            weather_data = get_weather_data(year, race)
            merged_results = merge_race_and_qualifying(race_results, qual_results)
            
            # Adding weather data to each driver's record
            for col in weather_data.columns:
                merged_results[col] = weather_data[col].iloc[0]
                
            merged_results['Year'] = year
            merged_results['RaceName'] = race
            all_data.append(merged_results)
    
    final_data = pd.concat(all_data, ignore_index=True)
    output_filename = f'{file_path}/f1_data_{start_year}_{end_year}.csv'
    final_data.to_csv(output_filename, index=False)
    print(f"Data preparation complete. File saved as '{output_filename}'.")

if __name__ == "__main__":
    # Example usage: Prepare data from 2013 to 2023
    data_path = './data'
    prepare_f1_data(2018, 2023, data_path)