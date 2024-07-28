import os
import pickle
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import inference, compute_model_metrics

# List of categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="session")
def data():
    """
    Fixture to load and provide the dataset.

    This fixture reads the census income dataset from the specified CSV file path
    and returns it as a Pandas DataFrame. The fixture has a session scope, meaning
    it is set up once per test session and shared across tests.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    data_path = "./data/census_income_clean.csv"
    df = pd.read_csv(data_path)
    return df

def test_column_names(data):
    """
    Test that the DataFrame contains all specified categorical feature columns.

    This test checks if all columns listed in `cat_features` are present in the
    DataFrame. It compares the actual columns in the DataFrame against the expected
    columns and raises an assertion error if any columns are missing.

    Args:
        data (pd.DataFrame): The DataFrame loaded by the `data` fixture.

    Raises:
        AssertionError: If any of the expected columns are missing in the DataFrame.
    """
    df_columns = data.columns

    # Determine which categorical features are present in DataFrame columns
    present_in_columns = [feature for feature in cat_features if feature in df_columns]
    
    # Assert that the result matches the expected features
    assert present_in_columns == cat_features, \
        f"Expected columns {cat_features} but got {present_in_columns}"

def load_model_artifacts(model_dir='./model'):
    """
    Load model, encoder, and label binarizer artifacts from disk.

    Args:
        model_dir (str): Directory where the model artifacts are stored.

    Returns:
        tuple: A tuple containing the loaded model, encoder, and label binarizer.
    """
    model_path = os.path.join(model_dir, 'trained_model.pkl')
    encoder_path = os.path.join(model_dir, 'encoder.pkl')
    lb_path = os.path.join(model_dir, 'labelizer.pkl')

    model = pickle.load(open(model_path, 'rb'))
    encoder = pickle.load(open(encoder_path, 'rb'))
    lb = pickle.load(open(lb_path, 'rb'))

    return model, encoder, lb

def process_test_data(test_data, encoder, lb):
    """
    Process the test data using the encoder and label binarizer.

    Args:
        test_data (pd.DataFrame): The test dataset.
        encoder: The encoder used for data transformation.
        lb: The label binarizer.

    Returns:
        tuple: Processed features (X_test) and labels (y_test).
    """
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    return X_test, y_test

def test_precision_and_recall(data):
    """
    Test the model's precision and recall metrics on the test dataset.

    This test evaluates the trained model's precision and recall metrics using the
    test dataset and asserts that these metrics meet the specified thresholds.

    Args:
        data (pd.DataFrame): The DataFrame loaded by the `data` fixture.

    Raises:
        AssertionError: If precision or recall falls below the defined thresholds.
    """
    # Split the data into training and testing sets
    _, test_data = train_test_split(data, test_size=0.20, random_state=42)

    # Load model and other artifacts
    model, encoder, lb = load_model_artifacts()

    # Process the test data
    X_test, y_test = process_test_data(test_data, encoder, lb)

    # Make predictions
    predictions = inference(model, X_test)

    # Compute metrics
    precision, recall, _ = compute_model_metrics(y_test, predictions)

    # Define thresholds for precision and recall
    precision_threshold = 0.70
    recall_threshold = 0.60

    # Assert that precision and recall meet the thresholds
    assert precision > precision_threshold, \
        f"Precision is less than {precision_threshold * 100}%"
    assert recall > recall_threshold, \
        f"Recall is less than {recall_threshold * 100}%"

def test_fbeta(data):
    """
    Test the model's F-beta score on the test dataset.

    This test evaluates the trained model's F-beta score using the test dataset and
    asserts that the score meets the specified threshold.

    Args:
        data (pd.DataFrame): The DataFrame loaded by the `data` fixture.

    Raises:
        AssertionError: If the F-beta score falls below the defined threshold.
    """
    # Split the data into training and testing sets
    _, test_data = train_test_split(data, test_size=0.20, random_state=42)

    # Load model and other artifacts
    model, encoder, lb = load_model_artifacts()

    # Process the test data
    X_test, y_test = process_test_data(test_data, encoder, lb)

    # Make predictions
    predictions = inference(model, X_test)

    # Compute metrics
    _, _, fbeta = compute_model_metrics(y_test, predictions)

    # Define threshold for F-beta
    fbeta_threshold = 0.60

    # Assert that F-beta meets the threshold
    assert fbeta > fbeta_threshold, \
        f"F-beta is less than {fbeta_threshold * 100}%"
