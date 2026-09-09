# Titanic Survival Prediction - Machine Learning Classification

A hands-on Machine Learning classification project based on the famous Titanic dataset. The project explores multiple classification algorithms to predict whether a passenger survived the Titanic disaster based on passenger and travel-related information.

The project demonstrates an end-to-end introductory Machine Learning workflow including:

**Data Loading → Data Exploration → Data Cleaning → Missing Value Handling → Categorical Encoding → Train-Test Split → Feature Scaling → Model Training → Prediction → Model Evaluation → Model Comparison**

---

## 🎯 Project Objective

The objective of this project is to build Machine Learning classification models that predict whether a Titanic passenger survived based on their passenger and travel-related information.

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| `0`   | Did not survive |
| `1`   | Survived        |

---

## 📊 Dataset

The Titanic dataset is loaded using Seaborn's built-in dataset:

```python
df = sns.load_dataset("titanic")
```

### Original Dataset

* **Rows:** 891
* **Columns:** 15

The dataset contains information such as:

* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Passenger fare
* Port of embarkation
* Whether the passenger was traveling alone
* Survival status

---

## 🧹 Data Cleaning & Preprocessing

### 1. Removing Unnecessary Features

The following columns were removed:

```python
df.drop(
    ["deck", "embark_town", "alive", "class", "who", "adult_male"],
    axis=1,
    inplace=True
)
```

These features were removed to simplify the feature set and eliminate redundant or derived information.

---

### 2. Handling Missing Age Values

The `age` column contained missing values.

Missing ages were replaced using the mean age:

```python
df["age"].fillna(df["age"].mean(), inplace=True)
```

After preprocessing, the `age` column contains no missing values.

---

### 3. Handling Missing Embarked Values

Rows containing missing values in the `embarked` column were removed:

```python
df.dropna(subset=["embarked"], inplace=True)
```

The dataset was reduced from:

**891 rows → 889 rows**

---

## 🔤 Categorical Encoding

Machine Learning algorithms require numerical input.

`LabelEncoder` was used to convert categorical variables into numerical values:

* `sex`
* `embarked`

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])
```

> Note: Numeric columns such as `age` and `fare` should retain their decimal values. Converting the entire dataframe using `astype(int)` is avoided because it unnecessarily truncates these continuous features.

---

## 🧩 Feature & Target Separation

The target variable `survived` was separated from the input features.

```python
X = df.drop("survived", axis=1)
y = df["survived"]
```

### Features Used

The model uses the following features:

* `pclass`
* `sex`
* `age`
* `sibsp`
* `parch`
* `fare`
* `embarked`
* `alone`

### Target

```text
survived
```

---

## ✂️ Train-Test Split

The dataset was divided into training and testing datasets using Scikit-learn's `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Configuration

| Parameter     | Value |
| ------------- | ----- |
| Test Size     | 20%   |
| Training Data | 80%   |
| Random State  | 42    |
| Test Samples  | 178   |

---

## 📏 Feature Scaling

Feature scaling was applied for algorithms that are sensitive to the scale of input features, such as KNN and SVM.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The test dataset is transformed using the scaler fitted on the training data to avoid data leakage.

---

# 🤖 Machine Learning Models

The project compares five classification algorithms.

### 1. K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
```

### Accuracy

**77.53%**

---

### 2. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Accuracy

**80.34%**

Logistic Regression provides a strong baseline for this binary classification problem.

---

### 3. Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

model_NB = GaussianNB()

model_NB.fit(X_train, y_train)

y_pred_NB = model_NB.predict(X_test)
```

### Accuracy

**77.53%**

---

### 4. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

model_DT = DecisionTreeClassifier(random_state=42)

model_DT.fit(X_train_scaled, y_train)

y_pred_DT = model_DT.predict(X_test_scaled)
```

### Accuracy

**76.97%**

---

### 5. Support Vector Classifier

```python
from sklearn.svm import SVC

model_svm = SVC(kernel="rbf")

model_svm.fit(X_train_scaled, y_train)

y_pred_svc = model_svm.predict(X_test_scaled)
```

### Accuracy

**82.58%**

Among the models tested in this notebook, SVM achieved the highest test accuracy.

---

# 📊 Model Comparison

| Model                         |   Accuracy |
| ----------------------------- | ---------: |
| K-Nearest Neighbors           |     77.53% |
| Logistic Regression           |     80.34% |
| Gaussian Naive Bayes          |     77.53% |
| Decision Tree                 |     76.97% |
| **Support Vector Classifier** | **82.58%** |

### 🏆 Best Performing Model

Based on the recorded test results:

**Support Vector Classifier (SVC) — 82.58% Accuracy**

---

# 🔲 Confusion Matrix

The confusion matrix for the Logistic Regression model was:

```text
[[90, 19],
 [16, 53]]
```

|          | Predicted 0 | Predicted 1 |
| -------- | ----------: | ----------: |
| Actual 0 |          90 |          19 |
| Actual 1 |          16 |          53 |

This represents:

* **90** correctly predicted non-survivors
* **53** correctly predicted survivors
* **19** non-survivors incorrectly predicted as survivors
* **16** survivors incorrectly predicted as non-survivors

---

# 📋 Logistic Regression Classification Report

| Class               | Precision | Recall | F1-Score |
| ------------------- | --------: | -----: | -------: |
| 0 — Did Not Survive |      0.85 |   0.83 |     0.84 |
| 1 — Survived        |      0.74 |   0.77 |     0.75 |

### Overall Metrics

| Metric           | Score |
| ---------------- | ----: |
| Accuracy         |  0.80 |
| Macro Average    |  0.79 |
| Weighted Average |  0.80 |

---

# 🧠 Machine Learning Workflow

```text
Titanic Dataset
       ↓
Data Exploration
       ↓
Remove Unnecessary Features
       ↓
Handle Missing Values
       ↓
Categorical Encoding
       ↓
Feature / Target Separation
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Train Multiple Classification Models
       ↓
Prediction
       ↓
Accuracy Evaluation
       ↓
Confusion Matrix
       ↓
Classification Report
       ↓
Model Comparison
```

---

# 🛠️ Technologies & Libraries

### Programming

* Python
* Jupyter Notebook

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Algorithms

* K-Nearest Neighbors
* Logistic Regression
* Gaussian Naive Bayes
* Decision Tree
* Support Vector Classifier

### Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

---

# 📂 Project Structure

```text
03-Titanic-Survival-Prediction/
│
├── Titanic.ipynb
└── README.md
```

---

# 📚 Concepts Learned

Through this project, I practiced:

* Binary Classification
* Dataset Exploration
* Data Cleaning
* Missing Value Handling
* Feature Selection
* Categorical Encoding
* Label Encoding
* Feature and Target Separation
* Train-Test Splitting
* Feature Scaling
* Model Training
* Model Prediction
* Model Evaluation
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Reports
* Comparing Multiple Classification Algorithms

---

# 💡 Key Takeaways

This project demonstrates how different Machine Learning classification algorithms perform on the same dataset.

Logistic Regression achieved **80.34% accuracy**, while the Support Vector Classifier achieved the highest recorded accuracy of **82.58%** among the models tested.

The project provides practical experience with the complete Machine Learning classification workflow:

```text
Data
  ↓
Cleaning
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Training
  ↓
Prediction
  ↓
Evaluation
  ↓
Model Comparison
```

---

# 🚀 Future Improvements

The project can be further improved by:

* Using One-Hot Encoding instead of Label Encoding
* Applying proper feature scaling through a Pipeline
* Using `ColumnTransformer` for preprocessing
* Performing cross-validation
* Hyperparameter tuning
* Optimizing the SVM model
* Comparing additional algorithms
* Handling class imbalance
* Performing feature importance analysis
* Improving feature engineering
* Creating visualizations for model comparison
* Building a reusable Scikit-learn Pipeline
* Saving the trained model using Joblib
* Creating a prediction API using FastAPI
* Containerizing the API using Docker
* Deploying the model as a web service

---

# 👨‍💻 Author

**Aditya Kumar Singh**

