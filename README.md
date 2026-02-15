**************************************************************
Problem Statement: Breast Cancer Diagnostic Classification
**************************************************************
This project implements a Machine Learning pipeline to classify breast cancer tumors as Malignant or Benign using the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

**************************************************************
Dataset description
**************************************************************
Source :
  > Source : kagglehub (yasserh/breast-cancer-dataset)
  > Source url : https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?select=breast-cancer.csv

Dataset Overview :
  > Total Sampels : 569
  > Total Features: 30 
  > Target: diagnosis (M = Malignant, B=Benign)

Implementaion Methodology:
  > Data preprocessing: Handled missing values and dropped irrelevant 'id' columns.
  > Scaling: Used StandardScaler to normalize the features to ensure distance-based models, such as KNN and logistic regression performs optimal.
  > 80% Training, 20% Testing with stratification.
  > Models Evaluated: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and XGBoost.

**************************************************************
Models Used and there perfromce comparsion table:
**************************************************************
Training on 30 features and 569 samples...

--- PERFORMANCE COMPARISON TABLE ---
| ML Model Name       |   Accuracy |      AUC |   Precision |   Recall |       F1 |      MCC |
|:--------------------|-----------:|---------:|------------:|---------:|---------:|---------:|
| Logistic Regression |   0.964912 | 0.996032 |    0.975    | 0.928571 | 0.95122  | 0.924518 |
| Decision Tree       |   0.929825 | 0.914683 |    0.947368 | 0.857143 | 0.9      | 0.848668 |
| kNN                 |   0.95614  | 0.982308 |    0.974359 | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes         |   0.921053 | 0.989087 |    0.923077 | 0.857143 | 0.888889 | 0.829162 |
| Random Forest       |   0.964912 | 0.99289  |    1        | 0.904762 | 0.95     | 0.92582  |
| XGBoost             |   0.973684 | 0.994048 |    1        | 0.928571 | 0.962963 | 0.944155 |


**********************************************************************************************
Observations on the performance of each model on the breast-cancer dataset
**********************************************************************************************
----------------------------------------------------------------------------------------------
ML Model Name    ->     Observation  
-----------------------------------------------------------------------------------------------
Logistic Regression -> Strong Linear Separability: Achieved a high accuracy of 96.49% and the highest AUC (0.9960). This indicates that the breast cancer features are highly linearly separable once standardized, making this a reliable baseline model.
-----------------------------------------------------------------------------------------------
Decision Tree -> While it performed well (92.11%), it had the lowest accuracy among the group. This suggests that a single tree might be slightly overfitting or missing the subtle patterns that ensemble methods captured.
-----------------------------------------------------------------------------------------------
kNN -> Effective Local Patterns: With 95.61% accuracy, kNN shows that similar tumor characteristics cluster well together. The high performance is a direct result of the StandardScaler ensuring all features contributed equally to distance calculations.
-----------------------------------------------------------------------------------------------
Naive Bayes -> Achieved 92.11%. Despite the naive assumption of feature independence, it remains competitive, though it slightly struggled with Recall (0.857) compared to more complex models.
-----------------------------------------------------------------------------------------------
Random Forest -> Balanced performance with 96.49% accuracy and a perfect Precision of 1.0. By averaging multiple trees, it eliminated the errors seen in the single Decision Tree, making it highly robust against false positives 
-----------------------------------------------------------------------------------------------
XGBoost -> The best overall model with 97.37% accuracy and 1.0 Precision. Its gradient boosting approach allowed it to minimize residuals effectively, capturing the complex relationships in the 30 features better than any other model.
-----------------------------------------------------------------------------------------------

OVERALL:

Top perfromer: XGBoost achieved the highest accuracy of 97.37%,while Random Forest and Logistic Regression holds at 96.49%.
precision: Random Forest and XGBoost achieved a perfect precision of 1.0, which is critical in medical diagnostics to avoid wrong labeling.
Stability: Logistic Regression also performed exceptionally well - 96.49%, suggesting the data has a strong linear seperation when scaled.

*******************************************************************************
How to Run 
*******************************************************************************
1. Clone the Repo: git clone https://github.com/2025aa05631-lab/ml_assignment_2.git
2. Install Dependencies: pip install -r requirements.txt
3. Run Training: python train_models.py
4. Launch App: streamlit run app_breastCancer.py

