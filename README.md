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

*******************************************************************************
Observations on the performance of each model on the breast-cancer dataset
*******************************************************************************
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

