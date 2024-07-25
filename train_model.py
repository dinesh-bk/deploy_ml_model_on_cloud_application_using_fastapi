import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slices, compute_confusion_matrix

# Initialize logging
logging.basicConfig(
    filename='journal.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def remove_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)

def load_data(filepath):
    return pd.read_csv(filepath)

def split_data(data, test_size=0.20):
    return train_test_split(data, test_size=test_size, random_state=42)

def process_train_test_data(train, test, cat_features, label):
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label=label, 
        training=False, encoder=encoder, lb=lb
    )
    return X_train, y_train, X_test, y_test, encoder, lb

def save_model(model, encoder, lb, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    filenames = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']
    objects = [model, encoder, lb]
    
    for filename, obj in zip(filenames, objects):
        with open(os.path.join(savepath, filename), 'wb') as f:
            pickle.dump(obj, f)
    
    logging.info(f"Model saved to disk: {savepath}")

def load_model(savepath):
    filenames = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']
    objects = []
    
    for filename in filenames:
        with open(os.path.join(savepath, filename), 'rb') as f:
            objects.append(pickle.load(f))
    
    return objects

def train_and_save_model(X_train, y_train, savepath):
    model, best_params = train_model(X_train, y_train)
    logging.info("Best Hyperparameters:")
    for param, value in best_params.items():
        logging.info(f"{param}: {value}")
    return model

def evaluate_model(model, X_test, y_test, lb):
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logging.info(f"Classification target labels: {list(lb.classes_)}")
    logging.info(f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")
    cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))
    logging.info(f"Confusion matrix:\n{cm}")
    return preds

def compute_slice_performance(test, cat_features, y_test, preds, slice_savepath):
    remove_if_exists(slice_savepath)
    for feature in cat_features:
        performance_df = compute_slices(test, feature, y_test, preds)
        performance_df.to_csv(slice_savepath, mode='a', index=False)
        logging.info(f"Performance on slice {feature}")
        logging.info(performance_df)

def main():
    # Configuration
    data_path = "./data/census_income_clean.csv"
    model_savepath = './model'
    slice_savepath = "./slice_output.txt"
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    # Load and split data
    data = load_data(data_path)
    train, test = split_data(data)

    # Process data
    X_train, y_train, X_test, y_test, encoder, lb = process_train_test_data(train, test, cat_features, "salary")

    # Train or load model
    if os.path.isfile(os.path.join(model_savepath, 'trained_model.pkl')):
        model, encoder, lb = load_model(model_savepath)
    else:
        model = train_and_save_model(X_train, y_train, model_savepath)
        save_model(model, encoder, lb, model_savepath)

    # Evaluate model
    preds = evaluate_model(model, X_test, y_test, lb)

    # Compute slice performance
    compute_slice_performance(test, cat_features, y_test, preds, slice_savepath)

if __name__ == "__main__":
    main()