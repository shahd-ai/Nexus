🩺 Health Insurance ML Project
📌 Overview

This project applies machine learning techniques to a health insurance dataset in order to address core business challenges in the insurance sector.
The notebook demonstrates how to predict charges, cluster clients, and detect fraud using real-world methods.

🎯 Business Objectives

BO1: Anticipate future medical claims costs to improve premium pricing and underwriting

BO2: Anticipate portfolio segmentation to design tailored coverage programs and strengthen client retention

BO3: Anticipate fraud exposure to safeguard reserves and reduce financial losses

📈 DSO (Days Sales Outstanding) – Technical Applications

DSO1: Predict insurance charges using regression modeling to estimate expected collection delays

DSO2: Monitor DSO across client clusters to anticipate payment behavior and optimize collections

DSO3: Integrate anomaly detection to link abnormal DSO patterns with potential fraud or disputed claims

🛠️ Key Steps

Data Understanding – Explore structure and variables

Data Exploration – Statistical and visual insights

Data Preparation – Cleaning and transformations

Feature Engineering – New features and encoding

Modeling – ML algorithms applied to each BO

Evaluation – Metrics to validate model performance

Deployment – Model ready for business integration

🤖 Algorithms Used

🌳 Random Forest → Claims cost prediction (Regression)

📊 KMeans Clustering → Client segmentation

🛡️ Isolation Forest → Fraud / anomaly detection

📂 Dataset

File: dataassurance.csv

Variables: Age, Sex, BMI, Children, Smoker, Region, Charges

📊 Evaluation Metrics

Regression: Mean Squared Error (MSE), R² Score

Clustering: Silhouette Score

Fraud Detection: Accuracy, Precision, Recall, F1-score

Financial: DSO evolution as performance indicator

📌 Visualization

The notebook includes visual analysis such as:

Histograms & Boxplots for distribution and outliers

Correlation heatmaps

Scatter plots highlighting anomalies with Isolation Forest

🚀 How to Run

Clone the repository

Install dependencies from requirements.txt

Place dataassurance.csv in the working directory

Run the notebook step by step
