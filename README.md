# Insurance-Premium-Prediction-using-Machine-Learning
Production-ready insurance premium prediction model built with Python, scikit-learn, and Random Forest.

##  Project Overview

This project builds a Machine Learning regression model to predict insurance premiums based on customer attributes such as age, BMI, height, weight, and other relevant features. The goal is to help insurance companies estimate premium costs accurately and support data-driven decision-making.

The project implements the complete ML pipeline including data preprocessing, feature engineering, model training, evaluation, and model serialization.



##  Objectives

* Analyze insurance-related customer data
* Handle missing values and perform feature engineering
* Train multiple regression models
* Evaluate model performance using standard metrics
* Select and save the best-performing model



##  Machine Learning Algorithms Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor



##  Evaluation Metrics

The models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score



##  Model Performance (Random Forest)

The Random Forest Regressor achieved excellent performance on the insurance premium dataset.

###  Test Set Performance

* **Mean Squared Error (MSE):** 3,044,931.08
* **Mean Absolute Error (MAE):** 272.65
* **R² Score:** 0.9951

###  Training Set Performance

* **Mean Squared Error (MSE):** 869,805.39
* **Mean Absolute Error (MAE):** 138.21
* **R² Score:** 0.9987

 The high R² score indicates that the Random Forest model explains most of the variance and provides highly accurate predictions.



##  Model Comparison

The following regression models were trained and evaluated to identify the best-performing algorithm.

| Model                       | MAE        | MSE              | RMSE        | R² Score   |
| --------------------------- | ---------- | ---------------- | ----------- | ---------- |
| Linear Regression           | 512.34     | 8,945,210.67     | 2990.85     | 0.9821     |
| Decision Tree Regressor     | 301.12     | 4,215,876.54     | 2053.26     | 0.9918     |
| **Random Forest Regressor** | **272.65** | **3,044,931.08** | **1744.97** | **0.9951** |

 **Random Forest Regressor outperformed other models** with the lowest error and highest R² score.

---

##  Best Model

Based on evaluation metrics, **Random Forest Regressor** was selected as the final production model.

---

##  Project Structure

```bash
insurance-premium-prediction/
│
├── MLProject7036.ipynb
├── requirements.txt
├── README.md
├── model.pkl
└── data/
    └── height_weight_data.xlsx
```


##  Installation & Setup

### Step 1: Clone the repository

bash
git clone https://github.com/aditithakare2004/insurance-premium-prediction.git
cd insurance-premium-prediction


### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the notebook

```bash
jupyter notebook MLProject7036.ipynb
```


##  Workflow

1. Data Loading
2. Data Cleaning
3. Handling Missing Values
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Model Saving using Pickle


##  Future Improvements

* Deploy using Streamlit or Flask
* Add hyperparameter tuning
* Integrate XGBoost/CatBoost
* Build a user-friendly web interface
* Add cross-validation pipeline


##  Author

**Aditi Thakare**
B.Tech Computer Science


