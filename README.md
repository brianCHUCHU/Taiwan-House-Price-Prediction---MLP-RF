# Taiwan House Price Prediction

This document presents a study on predicting house prices in Taiwan, covering preprocessing techniques, model selection, results, and evaluation.

## Quick Links

- [Detailed Analysis Paper](https://github.com/brianCHUCHU/Taiwan-House-Price-Prediction---MLP-RF/blob/main/Data_Mining_Homework_1.pdf): Detailed analysis and insights are shown in the pdf.

## Environment Setup
- To run the provided code, use Kaggle Notebook or Colab (with necessary file path modifications) to utilize CUDA (GPU) for improved efficiency.
- Two adjustable parameters: `target_y` and `layer_name`. `target_y` can be set as either 'log_price' or 'Price' for label type, and `layer_name` can be assigned 1 or 2 (theoretically could be more, but the number of layers is randomly generated in this notebook); `layer_name = 1` indicates that the MLP model is not used.

## Preprocessing
### Feature Engineering
- Transformation into polar coordinates for location features to establish a high correlation with house prices.
- Aggregation of certain features for improved continuity and representativeness.
- Addition of categorical features to represent the absence of certain continuous features.
- Removal of duplicated or useless features.

### Outlier Removal
- Removal of outliers by grouping districts and discarding rows that exceed 1.5 standard deviations.

### Normalization
- Transformation of the target variable 'Price' using `numpy.log1p()` and z-score normalization.
- Z-score normalization of all numeric columns.

## Models
### Model Construction
- Chosen model: Random Forest, an ensemble learning method considering interactions between columns.

### Hyperparameters Tuning
- Bayesian Optimization for Random Forest, XGBoost, and CatBoost.
- Random search for Multi-Layer Perceptron (MLP) due to its complex architecture and hyperparameters.

## Results Interpretation
### Model Performance
- Cross-validation (cv = 3) used to evaluate training performance.
- Negative mean absolute error for the three folds reported.

### Empirical Results
- Y Transformation: Validation scores suggest that 'log_price' typically does not outperform 'Price'.
- Outlier Removal: Not executed in the provided code due to increased overfitting tendencies.
- Model Selection: MLP's performance subpar, XGBoost and CatBoost underperformed compared to Random Forest.

### Conclusion
- Promising avenues for future research include dynamically adjusting computational resources, investigating reasons for MLP failures, and exploring more extensive feature engineering.

## References
- Kumar, S., Bhatt, S., Phudinawala, H. (2023) "Predicting House Price with Deep learning: A comparative study of Machine Learning Models."
