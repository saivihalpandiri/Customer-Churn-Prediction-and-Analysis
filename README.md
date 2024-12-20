# Customer-Churn-Prediction-and-Analysis
# 1. Data Preparation
 # 1.1 Dataset Overview
Description:
The dataset consists of customer demographic data, service details, and churn behavior. Key
columns include:
CustomerID: Unique identifier for each customer.
Gender: Categorical variable indicating the customer's gender.
Age: Numeric value representing the age of the customer.
Tenure: Length of customer engagement with the service in months.
MonthlyCharges and TotalCharges: Monetary attributes related to service usage.
PaymentMethod: The mode of payment used.
Churn: Binary target variable indicating whether a customer churned (Yes) or not (No).
# 1.2 Missing Value Handling
Missing Values Identified:
The TotalCharges column had missing values. The approach taken was to drop these rows for
simplicity and to avoid imputing values, which could introduce bias.
Impact:
A small fraction of rows was removed, which is unlikely to affect model performance given
the modest dataset size.
(Example: Use missing_values.plot(kind='bar').)
1.3 Data Type Adjustments
Conversion to Numeric:
The TotalCharges column was converted to a numeric data type using error coercion. Any
non-convertible values were treated as missing and removed subsequently.
A before-and-after summary of data types using df.info().
# 2. Exploratory Data Analysis (EDA)
# 2.1 Univariate Analysis
Churn Distribution:
A count plot was generated to examine the proportion of churned vs. non-churned customers.
Insight: The data showed class imbalance, with more non-churners than churners.
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.show()
# 2.2 Bivariate Analysis
Monthly Charges vs. Churn:
A boxplot compared MonthlyCharges across churn categories. It revealed whether customers
with higher/lower charges were more likely to churn.
Insight: Customers with higher MonthlyCharges showed a greater propensity to churn.
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn')
plt.show()
# 3. Feature Engineering
# 3.1 Label Encoding
Transformation Process:
Gender, PaymentMethod, and Churn were converted to numeric values using a label encoder.
This step ensured compatibility with machine learning algorithms.
Example Encoding:
Gender: Male → 0, Female → 1
PaymentMethod: Encoded as distinct integers for each payment type.
3.2 Feature Scaling
Why Scaling?
Features like Age, MonthlyCharges, and TotalCharges varied in magnitude. StandardScaler
normalized these features to ensure all variables contributed equally during model training.
# 4. Predictive Modeling
# 4.1 Model Selection
Models used: Logistic Regression, Decision Tree, Random Forest. Each model was trained
using an 80-20 train-test split.
Metrics Evaluated: Accuracy, Precision, Recall, and F1-Score.
Observation:
Random Forest outperformed other models, achieving an accuracy of 75%. However,
precision and recall were low for the minority class (churners).
metric_df = pd.DataFrame(results).T
metric_df.plot(kind='bar')
plt.title('Model Performance Comparison')
plt.show()
# 4.2 Feature Importance in Random Forest
Random Forest provides insights into which features contributed most to predictions. The top
features were likely MonthlyCharges, Tenure, and ServiceUsage1.
importances = best_model.feature_importances_
plt.barh(X.columns, importances)
plt.title('Feature Importances')
plt.show()
# 5. Recommendation Engine
# 5.1 Approach
The recommendation engine was based on cosine similarity across ServiceUsage1,
ServiceUsage2, and ServiceUsage3.
How It Works:
Computes pairwise similarity between customers based on service usage patterns.
Returns the most similar customer for personalized recommendations.
# 5.2 Output Example
For a given customer, the engine suggests a similar user for marketing or retention strategies.
python
sns.heatmap(similarity, cmap='viridis')
plt.title('Cosine Similarity Heatmap')
plt.show()
# 6. Recommendations and Next Steps
Data Imbalance Handling:
Consider SMOTE (Synthetic Minority Over-sampling Technique) to balance churn classes.
Model Optimization:
Hyperparameter tuning of Random Forest could improve performance.
Advanced Techniques:
Try Gradient Boosting models like XGBoost or LightGBM for better prediction.
Implement ensemble methods for robust predictions.
Enrich Data:Incorporate additional features like geographical data or customer reviews.
Refine Recommendation Engine:
## Overview
The analysis applied several machine learning models to predict customer churn. Each model
was evaluated based on its performance metrics, primarily accuracy. The table below
summarizes the key findings:
# Machine Learning Models Applied Accuracy
Random Forest 75.00%
Logistic Regression 76.25%
Decision Tree 74.50%
Naive Bayes Gaussian 72.40%
XGB Classifier 80.86%

# Model Precision
Machine Learning Models Applied Precision
Random Forest 60.00%
Logistic Regression 63.50%
Decision Tree 58.75%
Naive Bayes Gaussian 55.40%
XGB Classifier 68.90%

# Model Recall
Machine Learning Models Applied Recall
Random Forest 75.00%
Logistic Regression 77.00%
Decision Tree 72.50%
Naive Bayes Gaussian 68.30%
XGB Classifier 79.10%
Machine Learning Models Applied Accuracy
Model F1-Score
Machine Learning Models Applied F1-Score
Random Forest 66.67%
Logistic Regression 69.80%
Decision Tree 64.80%
Naive Bayes Gaussian 61.20%
XGB Classifier 73.90%
