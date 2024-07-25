import os
import pickle
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from .ml.data import process_data
from .ml.model import inference, compute_model_metrics

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

def test_accuracy(data):
    """
    Test the model's performance metrics on the test dataset.

    This test evaluates the trained model's performance using the test dataset. It
    loads the model, encoder, and label binarizer from disk, processes the test data,
    makes predictions, and calculates precision, recall, and F-beta metrics. It then
    asserts that these metrics meet the specified thresholds.

    Args:
        data (pd.DataFrame): The DataFrame loaded by the `data` fixture.

    Raises:
        AssertionError: If any of the metrics fall below the defined thresholds.
    """
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

    # Define paths to model artifacts
    model_dir = './model'
    model_filename = 'trained_model.pkl'
    encoder_filename = 'encoder.pkl'
    lb_filename = 'labelizer.pkl'

    # Load model and other artifacts
    model_path = os.path.join(model_dir, model_filename)
    encoder_path = os.path.join(model_dir, encoder_filename)
    lb_path = os.path.join(model_dir, lb_filename)

    model = pickle.load(open(model_path, 'rb'))
    encoder = pickle.load(open(encoder_path, 'rb'))
    lb = pickle.load(open(lb_path, 'rb'))

    # Process the test data
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make predictions
    predictions = inference(model, X_test)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    # Define thresholds for metrics
    precision_threshold = 0.70
    recall_threshold = 0.60
    fbeta_threshold = 0.60

    # Assert that metrics meet the thresholds
    assert precision > precision_threshold, \
        f"Precision is less than {precision_threshold * 100}%"
    assert recall > recall_threshold, \
        f"Recall is less than {recall_threshold * 100}%"
    assert fbeta > fbeta_threshold, \
        f"F-beta is less than {fbeta_threshold * 100}%"
