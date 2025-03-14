# Medical Malpractice Compensation Prediction

## Summary
This project focuses on predicting medical malpractice compensation, addressing the challenges posed by a complex dataset with a significant number of outliers. The methodology involved structured data management using DVC, GitHub, and a Cookiecutter-based infrastructure, along with preprocessing techniques like target transformations (logarithmic, square root, Box-Cox) and outlier handling. A variety of machine learning models, including Linear Regression, KNN, Random Forest, and XGBoost, were evaluated. XGBoost emerged as the best-performing model due to its robustness against outliers, ability to capture subtle patterns, and superior speed compared to Random Forests. Preprocessing played a pivotal role in shaping model performance, with square root and logarithmic transformations yielding better generalization. The project underscores the importance of ethical and accurate compensation prediction for enhancing justice and improving insurance risk management. The results demonstrate that careful preprocessing significantly influences model effectiveness, particularly for datasets with high variability and extreme values.

## Literature Review
According to a recent study published in the US News and World Report, the cost of medical malpractice in the United States is $55.6 billion a year, which is 2.4 percent of annual health-care spending. Another 2011 study published in the New England Journal of Medicine revealed that annually, during the period 1991 to 2005, 7.4% of all physicians licensed in the US had a malpractice claim. These staggering numbers not only contribute to the high cost of health care, but the size of successful malpractice claims also contributes to high premiums for medical malpractice insurance.

## Introduction
### Handling Outliers and Modeling Challenges
Data on medical malpractice compensation is very complex, containing a large number of outliers (exceptional cases with very large compensation values) due to the objective nature of compensation decisions. In fact, the compensation value does not depend on fixed criteria but is affected by a wide range of factors, some of which are not available in the current dataset. This creates a major challenge in data processing and building an accurate machine learning model, so the main task is to reduce the influence of outliers that can bias the results of normal cases, which represent most of the existing data.

### The Importance of This Work
Accurate prediction of the value of medical malpractice compensation is of paramount importance because it contributes to achieving justice for the injured, ensuring that each individual receives appropriate compensation. In addition, this model can help improve risk management for insurance companies and reduce the costs associated with ill-considered decisions. Addressing this issue is not only an economic issue, but also an ethical one that aims to support the transparency of the compensation system and ensure the rights of all.

## About Dataset
The data set contains information about the last 79210 claim payments made.
- **Amount**: Amount of the claim payment in dollars
- **Severity**: The severity rating of damage to the patient, from 1 (emotional trauma) to 9 (death)
- **Age**: Age of the claimant in years
- **Private Attorney**: Whether the claimant was represented by a private attorney
- **Marital Status**: Marital status of the claimant
- **Specialty**: Specialty of the physician involved in the lawsuit
- **Insurance**: Type of medical insurance carried by the patient
- **Gender**: Patient Gender

## Methodology
### Approach:
This project adopted a comprehensive and structured approach to predict medical malpractice compensation. The methods used include data management, preprocessing, model development, and performance evaluation. The key techniques and algorithms applied are as follows:

1. **Data Management & Project Infrastructure**:
   - **DVC (Data Version Control)**: Utilized for managing dataset versions, ensuring data consistency and reproducibility across different stages of the project.
   - **GitHub**: Employed for code version control, enabling the tracking of changes and efficient collaboration within the development team.
   - **Project Infrastructure**: Structured using the Cookiecutter template, which helps in organizing the code and streamlining machine learning workflows.
   - **DVCLive**: Incorporated to log and track the training process, providing real-time monitoring of model performance, hyperparameter tuning, and experiment progress.

2. **DVC Pipeline**:
   The project adopted a DVC pipeline to organize and streamline the machine learning workflow. The pipeline includes the following stages:
   - Data Collection
   - Data Preprocessing
   - Feature Engineering
   - Model Building
   - Model Evaluation

3. **Data Preprocessing**:
   - The dataset underwent Exploratory Data Analysis (EDA) to gain insights into its structure, distribution, and identify missing values or outliers.
   - Several preprocessing techniques were applied, including:
     - **Target Encoding, One-Hot Encoding, and Label Encoding**: To convert categorical variables into numerical format.
     - **Outlier Handling**: Removing Outliers within a threshold, and transformations such as Log2, Square Root, Winsorization, Robust Scaler, Box-Cox, and Quantile Transformation were applied to the target variable to mitigate the effect of extreme outliers.

4. **Model Development**:
   - A combination of linear and ensemble models was used for prediction:
     - **Linear Regression and Polynomial Regression**
     - **ElasticNet, Ridge, and Lasso Regularization models**
     - **RandomForest Regression and XGBoost Regression**
     - **Ensemble Regressors**
   - Hyperparameter tuning was carried out using Randomized Search and GridSearch.

5. **User Interface**:
   - A Flask API was used to create a simple and efficient backend for model deployment, and to provide predictions through the user interface.

### Evaluation Criteria:
To evaluate the model's performance, the following metrics were used:
- **R² (Coefficient of Determination)**: The first criterion used in model evaluation processes and examines the model's ability to capture complex patterns in the data. It was used in almost all experiments as it gave a more general impression and made it easier to assess the presence of overfitting when comparing the result of the test data and the training data. However, it is not sufficient to judge the model's performance in generalization, especially in cases of excessive distortion of the data from the transformation processes to the target.
- **RMSE (Root Mean Squared Error)**: It gives more weight to larger errors, it helps identify large discrepancies in predictions that may be more problematic in high-value cases, such as large medical malpractice claims. It is particularly useful when large errors need to be penalized more heavily, which is often the case in high-cost insurance scenarios.
- **MAE (Mean Absolute Error)**: Unlike RMSE, it does not emphasize larger errors, providing a more straightforward and interpretable measure of overall accuracy. It is valuable in evaluating the model’s ability to make consistent predictions without being overly sensitive to outliers or large errors, ensuring that typical claims are predicted accurately.
- **Predicted vs. Actual Values Diagram**: This visual Plot allows for an immediate understanding of how well the model’s predictions match the actual values. It helps identify patterns such as bias or systematic deviations in predictions, particularly useful for detecting if the model is overestimating or underestimating certain types of claims or even data distortions caused by hard-transformations that mentioned earlier.

## Results
The focus of this study was more on data preprocessing techniques than on parameter tuning, as preprocessing plays a crucial role in shaping the performance of the models. Although parameter tuning improved performance, its impact was relatively minor compared to the effect of preprocessing. XGBoost emerged as the most extensively tested model due to its exceptional robustness against outliers and its ability to capture subtle patterns effectively. It also demonstrated a significant speed advantage over random forests, being approximately ten times faster. As a result, XGBoost became the primary model for this project, with significant emphasis placed on exploring different preprocessing techniques within its framework.

Below is a summary of the experiments conducted across several models using key metrics:

| Model                        | R² (Train) | R² (Test) | MAE       | RMSE (Test)       |
|------------------------------|------------|-----------|-----------|-------------------|
| Linear Regression            | 0.2981     | 0.2967    | 113668.9094 | Not provided      |
| KNN                          | 0.6825     | 0.6157    | 73749.9137 | Not provided      |
| Decision Tree                | 0.8258     | 0.4989    | Not provided | 137250.4649      |
| Random Forest                | 0.7964     | 0.5464    | 76866.4910 | Not provided      |
| Ensemble (KNN + RF + XGB)    | 0.6864     | 0.6077    | 75063.8808 | Not provided      |
| Polynomial (Degree 2,3)      | 0.4        | 0.45      | Not provided | Not provided      |
| XGBoost                      | 0.7011     | 0.6605    | 70102.2202 | Not provided      |
| XGBoost (log2 transform)     | 0.6028     | 0.5507    | 1.783     | Not provided      |
| XGBoost (sqrt transform)     | 0.6894     | 0.6492    | 7,639.73  | Not provided      |
| XGBoost (winsorization)      | 0.7761     | 0.7414    | Not provided | Not provided      |

## Discussion
### Preprocessing
- **Target Encoding**: The medical specialty column contained over 20 unique values, making one-hot encoding impractical. Target encoding was initially applied, yielding strong results. However, it caused data leakage, resulting in overly optimistic and unreliable performance, so it was abandoned, and got replaced by one-hot encoding.
- **Handling Missing Data**: Missing values in the insurance type and marital status columns were initially replaced with their most frequent values. This approach led to poor results as it disrupted data patterns. A better strategy was to treat missing data as a separate category, which provided meaningful patterns for ensemble models and significantly improved performance.
- **Removing Outliers**: The extreme values in this dataset constitute about 35,000 samples, which represent at least half of the total data. Any attempt to remove these extreme values resulted in significantly poor performance in terms of generalization across almost all trained models, as the data lost crucial patterns during the removal process. As a result, this approach was ultimately excluded from the analysis.
- **Winsorization**: Winsorizing the target data seemed beneficial initially, as it reduced the influence of extreme outliers and improved the coefficient of determination (R²). However, further analysis using metrics like MAE and RMSE revealed that this improvement was superficial. The predicted vs. actual plots exposed odd patterns, indicating the model struggled to generalize effectively.
- **Logarithmic Transformation**: Applying a log2 transformation balanced the target distribution and significantly reduced MAE. However, its lower R² (around 55%) indicated a weaker ability to capture complex patterns, which hindered model generalization.
- **Square Root Transformation**: The square root transformation proved effective by slightly reducing the influence of outliers while maintaining better overall performance across metrics, particularly in generalization with improved MAE and RMSE.
- **Feature Engineering**: It was observed that the performance of the models declined significantly after any feature identification process, often to the extent that even the most powerful models failed to fit the data properly. Consequently, this approach was excluded. Additionally, the dataset did not contain many features, which limited the potential for meaningful feature extraction. Although attempts were made to extract new features, these efforts did not yield significant improvements and only increased model complexity without noticeable results. Therefore, the focus shifted towards target transformation processes and adjusting model parameters, rather than concentrating on feature extraction or identification.
- **Box-Cox Transformation**: Although Box-Cox transformations mitigated the impact of outliers, they did not emerge as the optimal preprocessing method for creating a generalized and robust model.

### Linear Regression
There were significant challenges in trying to fit linear regression models in general, even regularized and polynomial models, as well as the special type Huber regression, but the results were poor and usually led to underfitting, which are expected and natural results given the behavior of the data and its need for models that are resistant to extreme values.

### Decision Trees
The use of clustering methods such as decision trees was the starting point for thinking about using clustering methods, due to their high resistance to handling extreme values, which constitute almost half of the dataset. However, decision trees alone were not sufficient, and with the presence of a huge amount of training data, the algorithm could not generalize a model with good performance. The result was over-generalization in all cases of parameter tuning or data cleaning.

### Random Forests
Random forests generally performed strongly and were previously used as the basic model for conducting experiments. They give results close to XGBoost but at a suitable fitting time that may reach more than ten times than XGBoost, so they were not relied upon primarily later, but they were sometimes used as evidence for comparison purposes.

### Ensemble Regressors
For the combined ensemble model, the outliers in the dataset appeared to have a significant impact, strong enough to override the collective decision of the three models. This may be due to the nature of the regression task, which involves predicting a numerical value rather than a probability. Although there was some hope for improvement with this approach, the influence of the outliers limited its effectiveness.

### KNN
The KNN model delivered surprisingly strong performance despite minimal tuning or in-depth exploration. However, the primary focus remained on XGBoost due to its superior performance in terms of fitting speed and its robustness in handling complex data patterns.

## Conclusion
- **Data Cleaning and Preprocessing**: The success of the model heavily relied on how well the data was cleaned and preprocessed, especially the transformation operations applied to the target column. Traditional metrics like R², RMSE, and MAE were valuable but did not fully reflect the performance, especially in cases of data distortion or extreme values.
- **Handling Extreme Values**: Ensemble regressor methods, while helpful, did not deliver satisfactory results due to their inability to manage extreme data behaviors. Removing extreme values, which comprised about 50% of the dataset, led to underfitting and poor performance across models. This demonstrated the need for strategies that address extreme values without discarding them.
- **Dual-Model Approach**: To address the issue of extreme data, a dual-model approach will be adopted. One model will be trained on extreme data (with serious issues) and another on balanced data (with normal issues). A pre-trained classifier will mediate between the two models, determining which model to use for predictions based on the sample's characteristics. This aims to mitigate the bias introduced by extreme data and enhance overall model accuracy.
- **Future Work and Refinement**: Future work will focus on refining the dual-model strategy and exploring additional techniques to handle extreme values. Moreover, advanced methods for detecting and managing data distortion will be explored to further enhance model performance.

## Resources
- **Dataset**: [Medical Malpractice Insurance Dataset](https://www.kaggle.com/datasets/gabrielsantello/medical-malpractice-insurance-dataset)
- **EDA**: [Medical Malpractice XGBoost Plotly](https://www.kaggle.com/code/gabrielsantello/medical-malpractice-xgboost-plotly)
- **Project’s Repository**: [MedicalClaimPredictor-ML](https://github.com/AhmadHudhud83/MedicalClaimPredictor-ML)

## By
 Ahmad Hudhud
