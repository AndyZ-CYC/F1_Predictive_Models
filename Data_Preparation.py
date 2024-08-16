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
# Circuit length mapping (in meters) for circuits used after 2017
CIRCUIT_LENGTHS = {
    "adelaide": 3780,
    "casablanca": 7618,  # Ain-Diab Circuit
    "aintree": 4828,
    "melbourne": 5278,  # Albert Park Circuit
    "portimão": 4653,  # Algarve International Circuit
    "estoril": 4360,  # Autódromo do Estoril
    "mexicocity": 4304,  # Autódromo Hermanos Rodríguez
    "riodejaneiro": 5031,  # Autódromo Internacional do Rio de Janeiro
    "mugello": 5245,  # Autodromo Internazionale del Mugello
    "imola": 4909,  # Autodromo Internazionale Enzo e Dino Ferrari
    "sãopaulo": 4309,  # Autodromo José Carlos Pace
    "monza": 5793,  # Autodromo Nazionale di Monza
    "buenosaires": 4259,  # Autódromo Oscar y Juan Gálvez
    "berlin": 8300,  # AVUS
    "sakhir": 5412,  # Bahrain International Circuit
    "baku": 6003,  # Baku City Circuit
    "westkingsdown": 4206,  # Brands Hatch Circuit
    "greaternoida": 5141,  # Buddh International Circuit
    "lemans": 4430,  # Bugatti Au Mans
    "paradise": 3650,  # Caesars Palace Grand Prix Circuit
    "saintgeneschampanelle": 8055,  # Charade Circuit
    "bern": 7208,  # Circuit Bremgarten
    "montmeló": 4657,  # Circuit de Barcelona-Catalunya
    "montecarlo": 3337,  # Circuit de Monaco
    "monaco": 3337,  # Circuit de Monaco
    "magnycours": 4411,  # Circuit de Nevers Magny-Cours
    "barcelona": 6316,  # Circuit de Pedralbes
    "gueux": 8302,  # Circuit de Reims-Gueux
    "spa-francorchamps": 7004,  # Circuit de Spa-Francorchamps
    "prenois": 3886,  # Circuit Dijon-Prenois
    "montréal": 4361,  # Circuit Gilles-Villeneuve
    "monttremblant": 4265,  # Circuit Mont-Tremblant
    "austin": 5513,  # Circuit of the Americas
    "lecastellet": 5842,  # Circuit Paul Ricard
    "zandvoort": 4259,  # Circuit Zandvoort
    "heusdenzolder": 4262,  # Circuit Zolder
    "porto": 7775,  # Circuito da Boavista
    "lisbon": 5440,  # Circuito de Monsanto
    "jerez": 4428,  # Circuito Permanente de Jerez
    "sansebastiandelosreyes": 3314,  # Circuito Permanente del Jarama
    "dallas": 3901,  # Dallas Fair Park
    "detroit": 4168,  # Detroit Street Circuit
    "castledonington": 4020,  # Donington Park
    "oyama": 4563,  # Fuji Speedway
    "hockenheim": 4574,  # Hockenheimring
    "budapest": 4381,  # Hungaroring
    "speedway": 4192,  # Indianapolis Motor Speedway
    "istanbul": 5338,  # Intercity Istanbul Park
    "jeddah": 6174,  # Jeddah Corniche Circuit
    "yeongam": 5615,  # Korea International Circuit
    "midrand": 4261,  # Kyalami Grand Prix Circuit
    "lasvegas": 6201,  # Las Vegas Strip Circuit
    "longbeach": 3275,  # Long Beach Street Circuit
    "lusail": 5419,  # Lusail International Circuit
    "marinabay": 4940,  # Marina Bay Street Circuit
    "singapore": 4940,  # Marina Bay Street Circuit
    "miami": 5412,  # Miami International Autodrome
    "montjuïc": 3791,  # Montjuïc circuit
    "bowmanville": 3957,  # Mosport International Raceway
    "nivellesbaulers": 3724,  # Nivelles-Baulers
    "nürburgring": 5148,  # Nürburgring
    "pescara": 25800,  # Pescara Circuit
    "phoenix": 3720,  # Phoenix Street Circuit
    "eastlondon": 3920,  # Prince George Circuit
    "spielberg": 4318,  # Red Bull Ring
    "morenovalley": 5271,  # Riverside International Raceway
    "orival": 6542,  # Rouen-Les-Essarts
    "anderstorp": 4031,  # Scandinavian Raceway
    "sebring": 8356,  # Sebring Raceway
    "sepang": 5543,  # Sepang International Circuit
    "shanghai": 5451,  # Shanghai International Circuit
    "silverstone": 5891,  # Silverstone Circuit
    "sochi": 5848,  # Sochi Autodrom
    "suzuka": 5807,  # Suzuka International Racing Course
    "mimasaka": 3703,  # TI Circuit Aida
    "valencia": 5419,  # Valencia Street Circuit
    "watkinsglen": 5430,  # Watkins Glen International
    "yasmarina": 5281,  # Yas Marina Circuit
    "yasisland": 5281,  # Yas Marina Circuit
    "zeltweg": 3186  # Zeltweg Airfield
}


def update_times(results):
    """
    Update the Time column in the race results DataFrame to reflect the complete race time for each driver.

    Parameters:
    results (pd.DataFrame): A DataFrame containing race results, including a Time column. The Time column
                            contains the complete time for the first-position driver and time differences
                            for other drivers.

    Returns:
    pd.DataFrame: The updated DataFrame where the Time column now contains the actual complete race time
                  for each driver.
    """
    if 'Time' not in results.columns or pd.isnull(results.iloc[0]['Time']):
        return results
    
    # Extract the time of the first-position driver
    base_time = results.iloc[0]['Time']
    
    # Update the Time column for all other drivers
    results['Time'] = results['Time'].apply(
        lambda x: base_time + x if pd.notnull(x) and x != base_time else x
    )
    return results


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
    # session.load()  # Load session data
    results = session.results
    race_date = session.date  # Get the race date
    total_laps = session.total_laps  # Get the total number of laps

    # Update the Time column to the actual complete time
    results = update_times(results)

    # Get the circuit length using the CIRCUIT_LENGTHS dictionary
    circuit_id = session.event['Location'].lower().replace(' ', '')
    track_length = CIRCUIT_LENGTHS.get(circuit_id, None)  # None if not found

    # Add the race data to the results DataFrame
    results['RaceDate'] = race_date  
    results['TotalLaps'] = total_laps
    results['LapLength'] = track_length
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
    return qual_results[['FullName', 'Position', 'Q1', 'Q2', 'Q3']]


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
    merged_results = pd.merge(race_results, qual_results, on='FullName', how='left', suffixes=('_Race', '_Qual'))
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