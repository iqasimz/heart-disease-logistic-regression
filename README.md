A Logistic Regression project to classify the presence of heart disease using the UCI Cleveland Heart Disease dataset.

Objectives:
- Predict heart disease (yes/no) from 13 clinical features
- Perform end-to-end data cleaning and preprocessing
- Standardize features for better model performance
- Train a Logistic Regression model
- Evaluate using Accuracy, Precision, Recall, F1, and ROC-AUC
- Interpret model coefficients to understand risk factors

Methodology:
- Used clean version of the UCI Cleveland Heart Disease dataset
- Verified no missing values in any column
- Split dataset into training (80%) and testing (20%) sets
- Scaled all features using `StandardScaler`
- Trained a `LogisticRegression()` model from Scikit-Learn
- Predicted outcomes and evaluated using multiple classification metrics
- Visualized Confusion Matrix and ROC Curve

Feature Coefficients:
Each logistic regression coefficient tells how a 1-unit increase in a feature affects the log-odds of heart disease:
- Positive Coefficient → increases probability of disease
- Negative Coefficient → decreases probability of disease
Interpreting coefficients helps identify risk factors like high `chol`, low `thalach`, or abnormal `oldpeak`.

Evaluation Metrics (Test Set):
Accuracy, Precision, Recall, F1 Score, ROC-AUC   
Model performs well with high ROC-AUC, indicating good distinction between disease and non-disease cases.

Visualizations:
- Confusion Matrix: shows TP, FP, TN, FN distribution
- ROC Curve: plots TPR vs FPR, with AUC = 0.84