
import fastf1
import pandas as pd
from fastf1 import utils

# Enable the cache for faster data retrieval
fastf1.Cache.enable_cache('cache_folder') 

def get_race_results(year, race_name):
    '''Fetch the race results for a specific race and year'''
    session = fastf1.get_session(year, race_name, 'R')
    session.load()
    results = session.results
    return results

def get_historical_results(start_year, end_year):
    '''Fetch race results from start_year to end_year'''
    all_results = []
    for year in range(start_year, end_year + 1):
        schedule = fastf1.get_event_schedule(year)
        for race in schedule['EventName']:
            results = get_race_results(year, race)
            # Include year and race name in the results
            results['Year'] = year
            results['RaceName'] = race
            all_results.append(results)
    return pd.concat(all_results, ignore_index=True)

def get_car_info(year, race_name):
    '''Fetch car/constructor information for a specific race'''
    session = fastf1.get_session(year, race_name, 'R')
    session.load()
    results = session.results
    return results[['Car', 'Constructor', 'ConstructorID', 'TeamColor']]

def get_additional_data(year, race_name):
    '''Fetch other relevant data such as qualifying results and weather conditions'''
    qualifying = fastf1.get_session(year, race_name, 'Q')
    qualifying.load()
    qual_results = qualifying.results
    return qual_results[['Driver', 'Position', 'Q1', 'Q2', 'Q3', 'LapTime']]

def prepare_f1_data():
    '''Main function to prepare F1 data for model training'''
    # Get race history from 2013 to 2023
    race_data = get_historical_results(2013, 2023)
    
    # Fetch additional data (for simplicity, just using the last race of each year here)
    additional_data = []
    for year in range(2013, 2023 + 1):
        last_race = fastf1.get_event_schedule(year).iloc[-1]['EventName']
        car_info = get_car_info(year, last_race)
        qual_results = get_additional_data(year, last_race)
        combined = pd.merge(car_info, qual_results, left_index=True, right_index=True, how='left')
        combined['Year'] = year
        combined['RaceName'] = last_race
        additional_data.append(combined)
    
    additional_data_df = pd.concat(additional_data, ignore_index=True)
    
    # Save the data
    race_data.to_csv('race_history_2013_2023.csv', index=False)
    additional_data_df.to_csv('additional_data.csv', index=False)
    print("Data preparation complete. Files saved as 'race_history_2013_2023.csv' and 'additional_data.csv'.")

if __name__ == "__main__":
    prepare_f1_data()
