# heart-diseases-predictor
Decision Tree vs Random Forest vs XGBoost

This project compares three powerful tree-based machine learning models on a real-world heart disease prediction dataset.

The goal is to understand the **performance trade-offs** between simple and ensemble tree classifiers in terms of accuracy, generalization, and interpretability.
 
Objective:
Build an end-to-end classification pipeline to:
- Predict whether a patient has heart disease
- Train and evaluate Decision Tree, Random Forest, and XGBoost
- Compare them using precision, recall, F1 score, ROC-AUC, and training time
- Visualize confusion matrices and ROC curves

Dataset:
**[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)**  

Pipeline Overview:
1. Data Preprocessing
   - Standardize features using `StandardScaler`
   - Train/test split (80/20)
2. Model Training
   - `DecisionTreeClassifier()`
   - `RandomForestClassifier(n_estimators=100)`
   - `XGBClassifier(n_estimators=100)`
3. Evaluation Metrics
   - Accuracy
   - Precision, Recall, F1 Score
   - ROC AUC
   - Confusion Matrix
   - ROC Curve
   - Training Time

Visual Outputs:
- Confusion matrix heatmaps
- ROC curve comparison
- Model metrics table

Insights:
	•	Decision Tree performed best overall, due to simplicity and small dataset.
	•	Random Forest showed strongest ranking performance (ROC AUC), but weaker prediction performance.
	•	XGBoost needs tuning to outperform others on small datasets.
	•	Tree models are great at capturing nonlinear interactions.
