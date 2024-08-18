import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def data_loader(file_path, columns_to_include, categorical_cols):
    """
    Load and preprocess the dataset by selecting specific columns, handling missing values,
    converting time-related columns to seconds, flagging missing values, and processing categorical variables

    Parameters:
    file_path (str): The path to the CSV file
    columns_to_include (list): List of columns to include in the final DataFrame
    categorical_cols (list): List of categorical columns

    Returns:
    pd.DataFrame: Preprocessed DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Select the specified columns
    df = df[columns_to_include]

    # drop rows with nan time
    df = df.dropna(subset=['Time'])
    
    # # Convert RaceDate to datetime
    # df['RaceDate'] = pd.to_datetime(df['RaceDate'])
    
    # Convert Q1_Qual, Q2_Qual, Q3_Qual to seconds
    for col in ['Q1_Qual', 'Q2_Qual', 'Q3_Qual']:
        df[col] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()
    
    # Flag missing values and fill them with a placeholder (e.g., max_time + 10)
    for col in ['Q1_Qual', 'Q2_Qual', 'Q3_Qual']:
        # Create a flag column
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        
        # Impute missing values with a high value
        max_time = df[col].max() if not np.isnan(df[col].max()) else 0
        df[col].fillna(max_time + 10, inplace=True)
    
    # Process categorical variables with one-hot encoding
    # categorical_cols = ['DriverId', 'TeamId', 'RaceName', 'Year']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df


def split_data(df, target_column):
    """
    Split the data into training and testing sets

    Parameters:
    df (pd.DataFrame): The preprocessed DataFrame
    target_column (str): The name of the y column

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model

    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training target

    Returns:
    RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set

    Parameters:
    model: Trained model
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test target

    Returns:
    None
    """
    predictions = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
    print("R^2 Score:", r2_score(y_test, predictions))


def main():
    # Define the columns to include in the dataset
    columns_to_include = ['DriverId', 'TeamId', 'GridPosition', 'Position_Qual', 'Q1_Qual', 
                          'Q2_Qual', 'Q3_Qual', 'AirTemp', 'Humidity', 'Pressure',
                          'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'Year',
                          'RaceName', 'TotalLength', 'Time']
    categorical_cols = ['DriverId', 'TeamId', 'RaceName', 'Year']

    filepath = './data/f1_data_processed.csv'

    df = data_loader(filepath, columns_to_include, categorical_cols)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df, 'Time')

    model_rf = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model_rf, X_test, y_test)

    # Get the feature importances and feature names
    importances = model_rf.feature_importances_
    features = X_train.columns

    # Create a DataFrame to view the feature importances
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print(importance_df.head(15))

    # plot the top 10 important features
    plot_df = importance_df.head(15)

    plt.figure(figsize=(10, 8))  # Increase figure size
    plt.barh(plot_df['Feature'], plot_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')

    # Adjusting font size for better readability
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.gca().invert_yaxis()  # Highest importance at the top

    plt.show()


if __name__ == "__main__":
    main()


