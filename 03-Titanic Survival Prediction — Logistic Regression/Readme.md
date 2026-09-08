# 🚢 Titanic Survival Prediction — Logistic Regression

A hands-on **Machine Learning Classification project** focused on predicting whether a passenger survived the Titanic disaster using **Logistic Regression**.

This project demonstrates a complete introductory classification workflow including **data exploration, data cleaning, missing-value handling, categorical encoding, train-test splitting, model training, prediction, and classification evaluation**.

---

## 🎯 Project Objective

The objective of this project is to build a Machine Learning model that predicts whether a Titanic passenger **survived or did not survive** based on passenger and travel-related information.

The project uses **Logistic Regression**, a fundamental classification algorithm for predicting binary outcomes.

### Target Variable

| Value | Meaning |
| :---: | :--- |
| 0 | Did not survive |
| 1 | Survived |

## 📊 Dataset
The Titanic dataset is loaded using Seaborn's built-in dataset:

```python
df = sns.load_dataset("titanic")
```

**Original Dataset**
*   **Rows:** 891
*   **Columns:** 15

The dataset contains information about passengers including:
*   Passenger class
*   Sex
*   Age
*   Number of siblings/spouses
*   Number of parents/children
*   Fare
*   Port of embarkation
*   Whether the passenger was alone
*   Survival status

## 🧹 Data Cleaning & Preprocessing
The dataset was cleaned before training the Machine Learning model.

### 1. Removing Unnecessary Features
The following columns were removed:

```python
df.drop(
    ["deck", "embark_town", "alive", "class", "who", "adult_male"],
    axis=1,
    inplace=True
)
```
These features were excluded to simplify the feature set and avoid redundant information.

### 2. Handling Missing Age Values
The age column contained missing values.
Missing ages were replaced using the mean age:

```python
df["age"].fillna(df["age"].mean(), inplace=True)
```
*After preprocessing:* Age → No missing values

### 3. Handling Missing Embarked Values
Rows with missing values in the `embarked` column were removed:

```python
df.dropna(subset=["embarked"], inplace=True)
```
*The dataset was reduced from:* 891 rows → 889 rows

## 🔤 Categorical Encoding
Machine Learning models require numerical input, so categorical variables were converted into numerical values.
LabelEncoder was used for:
*   `sex`
*   `embarked`

Implementation:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])
```

The final dataframe was converted to integer type:

```python
df = df.astype(int)
```

## 🧩 Feature & Target Separation
The target variable `survived` was separated from the input features.

```python
X = df.drop("survived", axis=1)
y = df["survived"]
```

**Features**
The model uses: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `alone`

**Target**
`survived`

## ✂️ Train-Test Split
The dataset was divided into training and testing datasets using Scikit-learn's `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

**Split Configuration**

| Parameter | Value |
| :--- | :--- |
| Test Size | 20% |
| Random State | 42 |
| Training Data | 80% |
| Testing Data | 20% |

The test set contains 178 passengers.

## 🤖 Machine Learning Model

**Logistic Regression**
The classification model used in this project is Logistic Regression.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

The trained model is then used to predict survival:

```python
y_pred = model.predict(X_test)
```

## 📊 Model Evaluation
The model was evaluated using:
*   Accuracy Score
*   Confusion Matrix
*   Classification Report

### 🎯 Accuracy
The Logistic Regression model achieved: **Accuracy = 80.34%**

```python
accuracy_score(y_test, y_pred)
```
*Result: 0.8033707865168539*

### 🔲 Confusion Matrix
The confusion matrix produced by the model was:

```python
[[90, 19],
 [16, 53]]
```

| | Predicted 0 | Predicted 1 |
| :--- | :---: | :---: |
| **Actual 0** | 90 | 19 |
| **Actual 1** | 16 | 53 |

This provides a breakdown of correct and incorrect survival predictions.

### 📋 Classification Report

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **0 — Did Not Survive** | 0.85 | 0.83 | 0.84 |
| **1 — Survived** | 0.74 | 0.77 | 0.75 |
| **Overall Accuracy** | | | **0.80** |

**Overall Metrics**
*   **Accuracy:**  0.80
*   **Macro Avg:** 0.79
*   **Weighted Avg:** 0.80

## 🧠 Machine Learning Workflow
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
Feature & Target Separation
       ↓
Train/Test Split
       ↓
Logistic Regression
       ↓
Prediction
       ↓
Accuracy Evaluation
       ↓
Confusion Matrix
       ↓
Classification Report
```

## 🛠️ Technologies & Libraries
**Programming**
*   Python
*   Jupyter Notebook

**Data Analysis**
*   Pandas
*   NumPy

**Visualization**
*   Matplotlib
*   Seaborn

**Machine Learning**
*   Scikit-learn
*   Logistic Regression
*   LabelEncoder
*   `train_test_split`
*   Accuracy Score
*   Confusion Matrix
*   Classification Report

## 📂 Project Structure
```text
03-Titanic-Survival-Prediction/
│
├── Titanic_logistic.ipynb
└── README.md
```


## 📚 Concepts Learned
Through this project, I practiced:
*   Binary Classification
*   Logistic Regression
*   Dataset exploration
*   Data cleaning
*   Missing-value handling
*   Feature selection
*   Categorical encoding
*   Label Encoding
*   Feature/target separation
*   Train-test splitting
*   Model training
*   Model prediction
*   Accuracy evaluation
*   Confusion Matrix
*   Precision
*   Recall
*   F1-score
*   Classification Report

## 💡 Key Takeaways
This project demonstrates that a relatively simple Logistic Regression model can learn useful patterns from passenger characteristics to predict Titanic survival.
The experiment also provides practical experience with the complete workflow required for a basic Machine Learning classification problem:

**Data → Cleaning → Preprocessing → Training → Prediction → Evaluation**

The final model achieved an 80.34% accuracy on the test dataset.

## 🚀 Future Improvements
This project can be extended by:
*   Comparing Logistic Regression with other classification algorithms
*   Trying Decision Tree
*   Trying Random Forest
*   Trying K-Nearest Neighbors
*   Trying Support Vector Machine
*   Using One-Hot Encoding instead of Label Encoding
*   Applying feature scaling
*   Performing cross-validation
*   Hyperparameter tuning
*   Comparing multiple evaluation metrics
*   Building a reusable Scikit-learn Pipeline
*   Deploying the trained model using FastAPI
*   Containerizing the prediction API with Docker

## 👨‍💻 Author
**Aditya Kumar Singh**
