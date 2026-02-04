# Customer Churn Prediction System

A complete end-to-end Machine Learning pipeline designed to predict customer churn using structured telecom data.
The project covers data preprocessing, feature engineering, model training, evaluation, and deployment through a Flask web application.

## Project Overview

Customer churn prediction is a classification problem aimed at identifying customers who are likely to discontinue a service.

### Dataset Overview

The dataset consists of 7,043 customer records with 23 features capturing demographic details, service subscriptions, billing information, and account tenure.
It includes both categorical and numerical variables such as contract type, internet service, monthly charges, total charges, SIM provider, and join year.
The target variable, Churn, indicates whether a customer discontinued the service, making the dataset suitable for supervised classification modeling.
This structured customer-level data supports comprehensive analysis of behavioral patterns influencing churn.



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
Implemented using structured Python modules to ensure a scalable and maintainable workflow.
The pipeline covers data preprocessing, feature engineering, model training, and evaluation in a systematic sequence.
Modular implementation enables reproducibility, performance optimization, and seamless integration with deployment environments.

 

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

Implemented multiple supervised learning algorithms to evaluate predictive performance and model suitability.
Compared models based on accuracy, precision, recall, F1-score, and ROC-AUC metrics to ensure objective selection.
The final model was chosen based on robustness, generalization capability, and alignment with business requirements.

 

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

Addressed class imbalance using appropriate resampling techniques to ensure fair model learning across target classes.
Applied methods such as oversampling and undersampling to improve minority class representation without distorting data integrity.
This approach enhanced model stability, reduced bias, and improved overall predictive performance, particularly for churn detection.



SMOTE is applied to address class imbalance before model training.


### 4. Missing Value Handling

Handled incomplete data using statistically appropriate imputation techniques based on feature characteristics and distribution.
Applied mean, median, or mode substitution and conditional imputation to preserve data consistency and minimize information loss.
This ensured data integrity, improved model reliability, and prevented performance degradation during training.

 

Supported imputation strategies:

* Mean
* Median
* Mode
* Forward Fill
* Backward Fill
* Random Sampling



### 5. Variable Transformation

Transformed variables to improve data distribution, reduce skewness, and enhance model interpretability.
Applied techniques such as normalization, standardization, and logarithmic scaling where appropriate.
These transformations improved algorithm convergence, reduced variance, and strengthened overall predictive performance.



Transformations applied:

* Log Transformation
* Square Root
* Standard Scaling
* Min-Max Scaling
* Power Transformation
* Quantile Transformation



### 6. Outlier Treatment

Identified anomalous observations using statistical techniques such as IQR and Z-score analysis.
Applied capping, transformation, or removal strategies based on the severity and business relevance of extreme values.
This process improved model robustness, minimized distortion in feature distributions, and enhanced overall predictive stability.



Winsorization (Gaussian capping) is applied to control extreme values.



### 7. Hyperparameter Tuning

Optimized model performance through systematic hyperparameter tuning using structured search strategies.
Applied techniques such as Grid Search and Random Search with cross-validation to identify optimal parameter combinations.
This process improved generalization capability, reduced overfitting, and enhanced overall predictive accuracy.

 

GridSearchCV is used to evaluate Logistic Regression parameters.



### 8. Logging Framework

Implemented a structured logging framework to monitor pipeline execution, model training, and evaluation stages.
Captured key events, performance metrics, and exception details to ensure traceability and operational transparency.
This enabled efficient debugging, performance tracking, and improved maintainability across development and deployment environments.

Custom logging system records:

* Execution flow
* Model metrics
* Errors
* Debug information



### 9. Web Application (Flask)

Developed a lightweight web application using Flask to serve the trained model through RESTful endpoints.
Integrated input validation, preprocessing logic, and real-time prediction capabilities within the application layer.
This enabled seamless deployment, user interaction, and scalable integration with external systems.
 

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

bash
pip install -r requirements.txt


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

python
main.py


Then execute:

bash
python main.py


Outputs:

* churn_model.pkl
* standard_scalar.pkl
* Log files



## Running the Web Application

bash
python app.py



Access in browser:


http://127.0.0.1:5000/






## Author

Ganesh Bachalakuri

