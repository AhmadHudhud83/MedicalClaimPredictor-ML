# Medical Malpractice Compensation Prediction

## Introduction
This project aims to predict medical malpractice compensation using machine learning techniques. The dataset is complex, featuring a large number of outliers due to the varied nature of compensation decisions. These outliers challenge accurate modeling, as they can skew predictions for typical cases. Our objective is to develop a robust model that minimizes the impact of extreme values while ensuring precise predictions.

Accurate compensation prediction is vital for delivering justice to claimants and enhancing risk management for insurance providers. Beyond its economic implications, this work addresses ethical concerns by promoting transparency in the compensation system.

**Context**: Medical malpractice in the U.S. costs over $55.6 billion annually, representing 2.4% of healthcare spending. High claim values also drive up malpractice insurance premiums, underscoring the need for reliable predictive models.

## Dataset
The dataset includes data from 79,210 claim payments with the following features:
- **Amount**: Claim payment in dollars
- **Severity**: Damage severity (1 = emotional trauma, 9 = death)
- **Age**: Claimant’s age
- **Private Attorney**: Presence of legal representation
- **Marital Status**: Claimant’s marital status
- **Specialty**: Physician’s medical specialty
- **Insurance**: Patient’s insurance type
- **Gender**: Patient’s gender

[Dataset Link](https://www.kaggle.com/datasets/gabrielsantello/medical-malpractice-insurance-dataset)

## Methodology
### Approach
- **Data Management**: Used `DVC` for dataset versioning, GitHub for code management, and a Cookiecutter template for project structure.
- **Preprocessing**: Applied encoding and transformations to address outliers.
- **Model Development**: Trained models including Linear Regression, `KNN`, Random Forest, and `XGBoost`, with hyperparameter tuning via Randomized Search and GridSearch.
- **Evaluation**: Assessed performance using R², RMSE, and MAE.

### Data Preprocessing
- **Encoding**: Converted categorical variables with one-hot and label encoding.
- **Outlier Handling**: Applied logarithmic, square root, and Box-Cox transformations to reduce outlier impact.
- **Missing Data**: Treated missing values as a separate category, boosting model performance.

### Model Development
- Focused on `XGBoost` for its robustness against outliers and faster training compared to Random Forest.
- Compared against Linear Regression, `KNN`, and Decision Trees.

### Evaluation Criteria
- **R²**: Assesses how well the model captures data patterns.
- **RMSE**: Emphasizes larger errors, critical for high-value claims.
- **MAE**: Measures overall accuracy simply and effectively.

## Results
Performance metrics for key models are summarized below:

| Model                      | R² (Train) | R² (Test) | MAE          | RMSE (Test)    |
|----------------------------|------------|-----------|--------------|----------------|
| Linear Regression          | 0.2981     | 0.2967    | 113,668.91   | -              |
| KNN                        | 0.6825     | 0.6157    | 73,749.91    | -              |
| Decision Tree              | 0.8258     | 0.4989    | -            | 137,250.46     |
| Random Forest              | 0.7964     | 0.5464    | 76,866.49    | -              |
| Ensemble (KNN + RF + XGB)  | 0.6864     | 0.6077    | 75,063.88    | -              |
| XGBoost                    | 0.7011     | 0.6605    | 70,102.22    | -              |
| XGBoost (log2 transform)   | 0.6028     | 0.5507    | 1.783        | -              |
| XGBoost (sqrt transform)   | 0.6894     | 0.6492    | 7,639.73     | -              |
| XGBoost (winsorization)    | 0.7761     | 0.7414    | - | - |

## Discussion
- **Preprocessing**: Square root and logarithmic transformations improved generalization, while winsorization showed superficial gains but poor real-world performance.
- **Missing Data**: Treating missing values as a distinct category outperformed imputation.
- **Outliers**: Removing outliers (nearly 50% of the data) led to underfitting; transformations proved more effective.
- **Model Performance**: `XGBoost` excelled due to its speed and resilience to outliers, outpacing other models like Random Forest.

## Conclusion
- **Key Finding**: Effective preprocessing, particularly target transformations, was essential for success. `XGBoost` with square root transformation offered the best results.
- **Future Work**: We plan to explore a dual-model approach—one for extreme cases and another for typical claims—guided by a classifier to enhance accuracy.

## Resources
- [Dataset](https://www.kaggle.com/datasets/gabrielsantello/medical-malpractice-insurance-dataset)
- [EDA Notebook](https://www.kaggle.com/code/gabrielsantello/medical-malpractice-xgboost-plotly)


**Author**: Ahmad Hudhud

