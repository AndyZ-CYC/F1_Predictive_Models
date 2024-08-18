import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def data_loader(filepath, columns_to_include, categorical_cols):
    """
    Load and preprocess the dataset by selecting specific columns and drop missing values

    Parameters:
    filepath (str): The path to the CSV file
    columns_to_include (list): List of columns to include in the output data
    categorical_cols (list): List of categorical columns

    Returns:
    pd.DataFrame: loaded DataFrame.
    """
    # load the dataset
    df = pd.read_csv(filepath)

    # select the specific columns
    df = df[columns_to_include]

    # drop na values
    df.dropna(inplace=True)

    # Apply one-hot encoding to categorical variables
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
    Train a RandomForestClassifier model

    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training target

    Returns:
    RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(random_state=42)
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
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))


def main():
    # Define the columns to include in the dataset
    columns_to_include = ['DriverId', 'TeamId', 'GridPosition', 'Year', 
                        'Position_Qual', 'AirTemp', 'Humidity', 'Pressure', 
                        'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 
                        'RaceName', 'Finished']
    categorical_cols = ['DriverId', 'TeamId', 'RaceName', 'Year']

    filepath = './data/f1_data_processed.csv'

    df = data_loader(filepath, columns_to_include, categorical_cols)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df, 'Finished')

    model_rf = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model_rf, X_test, y_test)

    # Get the feature importances and feature names
    importances = model_rf.feature_importances_
    features = X_train.columns

    # Create a DataFrame to view the feature importances
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print(importance_df.head(10))

    # plot the top 10 important features
    plot_df = importance_df.head(10)

    plt.figure(figsize=(10, 8))  # Increase figure size
    plt.barh(plot_df['Feature'], plot_df['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')

    # Adjusting font size for better readability
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()  # Adjust layout to prevent clipping

    plt.show()


if __name__ == "__main__":
    main()