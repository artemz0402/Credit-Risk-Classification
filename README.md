Lending companies lend money or properties to borrowers with the expectation that the borrower will either return the asset or repay the lender. Credit risk is associated with a borrower not returning an asset or paying back a loan, causing the lender to lose money. In this analysis, I used machine learning to analyze a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

Using a machine learning model, I aimed to determine which loans are healthy (low-risk) or non-healthy (high-risk) based on the loan status provided by the lending company.

The logistic regression algorithm was selected for our machine learning model due to its effectiveness in predicting the probability of a target variable in classification problems.

The dataset provided by the lending company contained financial information related to loans, including variables such as loan amount, interest rate, borrower income, and credit score. The target variable, loan status, was imbalanced, with a higher number of healthy loans (0) compared to high-risk loans (1). The value counts for the target variable were: 18,765 healthy loans and 619 high-risk loans.

We followed several stages in the machine learning process:

Data Preprocessing: Cleaning and preparing the data for analysis, including handling missing values and normalizing features.
Model Selection: Choosing logistic regression and other algorithms for evaluation.
Model Training: Training the models on the dataset.
Model Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
Model Tuning: Fine-tuning the models to improve performance.
Results
Logistic Regression Model fitted with Imbalanced Data:
Accuracy: 99%
Precision (Healthy loans): 1.00
Recall (Healthy loans): 0.99
Precision (High-risk loans): 0.84
Recall (High-risk loans): 0.99
F1-Score (Healthy loans): 1.00
F1-Score (High-risk loans): 0.91
According to the confusion matrix for the imbalanced data:

Out of the 18,765 healthy loans, the model predicted 18,663 correctly and 102 incorrectly.
Out of the 619 high-risk loans, the model predicted 563 correctly and 56 incorrectly.
Logistic Regression Model fitted with Balanced (Oversampled) Data:
Accuracy: 99%
Precision (Healthy loans): 1.00
Recall (Healthy loans): 0.99
Precision (High-risk loans): 0.84
Recall (High-risk loans): 0.99
F1-Score (Healthy loans): 1.00
F1-Score (High-risk loans): 0.91
Using the RandomOverSampler module to balance the dataset, the value counts for the oversampled data were: 56,271 healthy loans and 56,271 high-risk loans.

According to the confusion matrix for the oversampled data:

Out of the 18,765 healthy loans, the model predicted 18,649 correctly and 116 incorrectly.
Out of the 619 high-risk loans, the model predicted 615 correctly and 4 incorrectly.
Summary
The logistic regression model fitted with oversampled data performed much better than the model fitted with imbalanced data, as the balanced dataset resulted in higher recall for high-risk loans, reducing false negatives. The model with balanced data achieved an accuracy score of 99%, with a significant improvement in recall for high-risk loans from 0.91 to 0.99.

A lending company would prefer a model that minimizes false positives (high-risk loans classified as healthy) and false negatives (healthy loans classified as high-risk). Based on this analysis, the logistic regression model fitted with balanced data is recommended. This model correctly classifies healthy and high-risk loans with fewer mistakes, providing a reliable tool for assessing credit risk.
