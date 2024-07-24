from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd 

def train_model(X_train, y_train):
    """
    Trains a machine learning model using GridSearchCV and returns the best model.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : LogisticRegression
        Trained machine learning model.
    """
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['liblinear', 'lbfgs'],  # Solver type
        'max_iter': [100, 200, 300]  # Maximum iterations
    }
    
    # Initialize the Logistic Regression model
    model = LogisticRegression()
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_
    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confusion matrix using the predictions and ground thruth provided
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    cm = confusion_matrix(y, preds)
    return cm



def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    ------
    df: 
        test dataframe pre-processed with features as column used for slices
    feature:
        feature on which to perform the slices
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized

    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """    
    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['feature','n_samples','precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = df[feature]==option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index(names='feature value', inplace=True)
    colList = list(perf_df.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df
