# Customer Churn Prediction System

A complete end-to-end Machine Learning pipeline designed to predict customer churn using structured telecom data.
The project covers data preprocessing, feature engineering, model training, evaluation, and deployment through a Flask web application.

## Project Overview

Customer churn prediction is a classification problem aimed at identifying customers who are likely to discontinue a service.

This project implements:

* Data cleaning and preprocessing
* Advanced feature transformation
* Outlier handling
* Class imbalance treatment
* Model comparison across multiple algorithms
* Final model selection and persistence
* Deployment using Flask

## Project Architecture


Customer Churn Prediction
│
├── Data Preprocessing
│   ├── Missing Value Handling
│   ├── Variable Transformation
│   ├── Outlier Handling
│   ├── Encoding
│   ├── Feature Selection
│   └── Correlation Filtering
│
├── Data Balancing (SMOTE)
│
├── Model Training & Evaluation
│   ├── KNN
│   ├── Naive Bayes
│   ├── Logistic Regression
│   ├── Decision Tree
│   ├── Random Forest
│   ├── XGBoost
│   └── SVM
│
├── Model Selection (AUC-ROC Comparison)
│
├── Model Persistence (Pickle)
│
└── Flask Web Deployment



## Core Modules

### 1. Model Training Pipeline

Implemented in:

📄 

The CHURN class performs:

* Dataset loading
* Train-test split
* Missing value treatment
* Feature transformations
* Outlier capping
* Encoding and feature selection
* SMOTE balancing
* Feature scaling
* Multi-model evaluation
* Final Logistic Regression model training
* Model serialization (churn_model.pkl)



### 2. Machine Learning Algorithms

📄 

Models implemented:

* K-Nearest Neighbors
* Gaussian Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Support Vector Machine

Each model logs:

* Accuracy
* Confusion Matrix
* Classification Report

AUC-ROC curves are plotted to compare performance.



### 3. Data Balancing

📄 

SMOTE is applied to address class imbalance before model training.


### 4. Missing Value Handling

📄 

Supported imputation strategies:

* Mean
* Median
* Mode
* Forward Fill
* Backward Fill
* Random Sampling



### 5. Variable Transformation

📄 

Transformations applied:

* Log Transformation
* Square Root
* Standard Scaling
* Min-Max Scaling
* Power Transformation
* Quantile Transformation



### 6. Outlier Treatment

📄 

Winsorization (Gaussian capping) is applied to control extreme values.



### 7. Hyperparameter Tuning

📄 

GridSearchCV is used to evaluate Logistic Regression parameters.



### 8. Logging Framework

📄 

Custom logging system records:

* Execution flow
* Model metrics
* Errors
* Debug information



### 9. Web Application (Flask)

📄 

Features:

* HTML-based UI
* Form-based customer input
* Real-time churn prediction
* Probability output (if supported)
* Model and scaler loading
* Feature alignment with trained model



## Installation

bash
git clone https://github.com/GaneshBachalakuri/churn-prediction.git
cd churn-prediction


Install dependencies:

```bash
pip install -r requirements.txt
```

Required Libraries:

* pandas
* numpy
* scikit-learn
* imbalanced-learn
* xgboost
* flask
* matplotlib
* seaborn
* feature-engine


## Model Training

Update dataset path inside:

```python
main.py
```

Then execute:

```bash
python main.py
```

Outputs:

* `churn_model.pkl`
* `standard_scalar.pkl`
* Log files

---

## Running the Web Application

```bash
python app.py
```


```Access in browser:


http://127.0.0.1:5000/```




## Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report
* ROC Curve
* AUC Score

Model comparison ensures selection of the most reliable classifier.



## Final Model

The final selected model:

* Logistic Regression
* Standard scaled features
* SMOTE-balanced training data
* Saved as serialized pickle object



## Business Impact

This system enables:

* Early identification of high-risk customers
* Proactive retention campaigns
* Reduced revenue loss
* Data-driven decision making



## Future Enhancements

* Deep Learning integration (ANN / LSTM)
* Real-time prediction API
* Docker containerization
* CI/CD integration
* Cloud deployment (AWS / Azure)



## Author

Ganesh Bachalakuri

