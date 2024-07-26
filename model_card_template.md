# Model Card

## Model Details

- **Model Type:** The model is a classification model, likely a Random Forest, trained to predict income categories based on various demographic and employment features.
- **Hyperparameters:** The model has been configured with the following hyperparameters:
  - `n_estimators`: 300
  - `min_samples_split`: 10
  - `min_samples_leaf`: 2
  - `max_features`: sqrt
  - `max_depth`: None

## Intended Use

The model has been developed to classify individuals into income categories, specifically predicting whether an individual's income exceeds $50K based on demographic and employment features.

## Training Data

The training data consists of demographic and employment information, including features such as workclass, education, and native country.

## Evaluation Data

The model has been evaluated on a test dataset with similar features to those used in training. Specific slices of data (e.g., based on workclass, education, and native country) have been analyzed to assess model performance across different demographic groups.

## Metrics

The model's performance has been evaluated using the following metrics:

- **Precision:** 0.783
- **Recall:** 0.625
- **F-beta Score:** 0.695

### Performance by Feature Slices

- **Workclass:** Performance metrics vary by workclass category, with precision ranging from 0.739 (State-gov) to 1.0 (Without-pay).
- **Education:** Performance metrics by education level show precision ranging from 0.728 (Some-college) to 1.0 (7th-8th, Doctorate).
- **Native Country:** The model's precision varies significantly by native country, with some categories (e.g., India, China) showing high precision and recall, while others (e.g., Laos, Peru) show lower or inconsistent performance.

## Ethical Considerations

The model should be used with caution, especially considering the potential for bias in predictions across different demographic groups. It is essential to ensure that the data used for training and evaluation is representative of the population and that the model does not disproportionately impact any particular group.

## Caveats and Recommendations

- **Bias:** Users should be aware of the potential for bias in the model's predictions, particularly related to demographic features like race, gender, or nationality.
- **Model Usage:** The model should not be the sole basis for making significant decisions, especially those that could impact individuals' livelihoods or well-being.
- **Continuous Monitoring:** Regular evaluation and updating of the model are recommended to ensure that it continues to perform well across diverse data and that any biases are identified and mitigated.