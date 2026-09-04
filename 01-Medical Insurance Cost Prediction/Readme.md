# 🏥 Medical Insurance Cost Prediction

A hands-on **Machine Learning Regression project** that predicts individual medical insurance charges based on demographic and lifestyle-related features.

This project focuses on the complete introductory ML workflow — from **data exploration and preprocessing to feature engineering, statistical analysis, feature selection, model training, and evaluation**.

---

## 🎯 Project Objective

The objective of this project is to build a Machine Learning model that can predict an individual's **medical insurance charges** using features such as:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The project also explores the relationships between different features and insurance charges using **correlation analysis and statistical hypothesis testing**.

---

## 📊 Dataset

The project uses the **Medical Cost Personal Dataset**, containing information about individuals and their corresponding medical insurance charges.

### Dataset Features

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `sex` | Gender |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependents |
| `smoker` | Smoking status |
| `region` | Residential region |
| `charges` | Medical insurance charges — target variable |

### Dataset Size

- **Rows:** 1,338
- **Features:** 7
- **Target:** `charges`

---

## 🔎 Exploratory Data Analysis

The project begins with exploratory analysis to understand the structure and characteristics of the dataset.

### EDA Includes

- Dataset shape
- Data types
- Descriptive statistics
- Missing-value analysis
- Duplicate-value detection
- Feature distributions
- Categorical feature analysis
- Correlation analysis

The analysis helps identify important patterns and relationships before building the model.

---

## 🧹 Data Preprocessing

The dataset was prepared for Machine Learning through several preprocessing steps.

### Preprocessing Performed

- Checked for missing values
- Identified and removed duplicate records
- Encoded categorical variables
- Renamed features for easier handling
- Scaled numerical features using `StandardScaler`

Categorical features such as:

- `sex`
- `smoker`
- `region`

were converted into numerical representations suitable for Machine Learning.

---

## 🧠 Feature Engineering

A new feature was created from BMI:

### BMI Category

BMI values were grouped into meaningful categories to explore whether different BMI ranges have a relationship with medical insurance charges.

This demonstrates how domain-related transformations can be used to create additional features from existing data.

---

## 📈 Statistical Analysis

Statistical techniques were used to understand feature relationships and support feature selection.

### Pearson Correlation

Pearson correlation was used to measure the linear relationship between numerical features and the target variable.

One of the strongest relationships observed was between:

```text
smoking status → insurance charges
```

The correlation between `is_smoker` and `charges` was approximately:

**0.787**

This indicates a strong positive linear relationship in this dataset.

### Chi-Square Test
Chi-square statistical testing was also used to investigate relationships involving categorical variables.
This helped provide an additional statistical perspective during feature selection.

---

## 🎯 Feature Selection
Feature relationships and statistical analysis were used to determine which variables were useful for predicting insurance charges.
The selected features were then used for model training.

---

## 🤖 Machine Learning Model

### Linear Regression
The primary model used in this project is: **Linear Regression**
Linear Regression was selected as an interpretable baseline model for predicting continuous insurance charges.

### Workflow
```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Categorical Encoding
     ↓
Feature Engineering
     ↓
Feature Scaling
     ↓
Correlation & Statistical Analysis
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Linear Regression
     ↓
Model Evaluation
```

---

## 📊 Model Performance
The trained Linear Regression model achieved:

| Metric | Score |
|---|---|
| R² Score | 0.8041 |
| Adjusted R² | 0.7988 |

### R² Score
An R² score of approximately 0.80 indicates that the model explains a substantial portion of the variation in medical insurance charges within the dataset.

### Adjusted R²
Adjusted R² was also calculated to account for the number of predictors used by the model.

---

## 🛠️ Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

## 📂 Project Structure
```text
01-Medical-Insurance-Cost-Prediction/
│
├── P1_insurance.ipynb
├── insurance.csv
└── README.md
```

---


## 💡 Key Learnings
Through this project, I practiced:

- Exploratory Data Analysis
- Data cleaning
- Duplicate handling
- Categorical encoding
- Feature engineering
- Feature scaling
- Pearson correlation
- Chi-square statistical testing
- Feature selection
- Train/test splitting
- Linear Regression
- R² evaluation
- Adjusted R² evaluation

The project strengthened the connection between data analysis, statistics, and Machine Learning model development.

---

## 🚀 Future Improvements
Possible improvements to this project include:

- Compare multiple regression algorithms
- Perform cross-validation
- Apply hyperparameter tuning
- Analyze residuals and model errors
- Build a complete preprocessing pipeline
- Experiment with regularization techniques such as Ridge and Lasso
- Deploy the trained model using FastAPI
- Containerize the application using Docker

---

## 👨‍💻 Author
**Aditya Kumar Singh**
