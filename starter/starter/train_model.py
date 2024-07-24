# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slices, compute_confusion_matrix
import pandas as pd
import os
import pickle
import logging

# Initialize logging
logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
def remove_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)

data = pd.read_csv("../data/census_income_clean.csv")
train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
savepath = '../model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

if os.path.isfile(os.path.join(savepath,filename[0])):
    model = pickle.load(open(os.path.join(savepath,filename[0]), 'rb'))
    encoder = pickle.load(open(os.path.join(savepath,filename[1]), 'rb'))
    lb = pickle.load(open(os.path.join(savepath,filename[2]), 'rb'))
else:
    model = train_model(X_train, y_train)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    pickle.dump(model, open(os.path.join(savepath, filename[0]), 'wb'))
    pickle.dump(encoder, open(os.path.join(savepath, filename[1]), 'wb'))
    pickle.dump(lb, open(os.path.join(savepath, filename[2]), 'wb'))
    logging.info(f"Model saved to disk: {savepath}")

# Evaluate trained model on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))
logging.info(f"Confusion matrix:\n{cm}")

# Compute performance on slices for categorical features
slice_savepath = "./slice_output.txt"
remove_if_exists(slice_savepath)

for feature in cat_features:
    performance_df = compute_slices(test, feature, y_test, preds)
    performance_df.to_csv(slice_savepath, mode='a', index=False)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)
