# 🚗 Car Price Prediction — Linear Regression

A hands-on **Machine Learning Regression project** focused on predicting used car prices from vehicle characteristics such as model, year, mileage, transmission, fuel type, tax, MPG, and engine size.

This project demonstrates a complete introductory Machine Learning workflow including **Exploratory Data Analysis, categorical encoding, feature scaling, train-test splitting, Linear Regression, and model evaluation**.

---

## 🎯 Project Objective

The objective of this project is to build a Machine Learning model that predicts the **price of a used Ford car** based on its available features.

The project also compares two different approaches for handling categorical variables:

- One-Hot Encoding
- Label Encoding

The resulting Linear Regression models are evaluated using **R² Score** and **Adjusted R² Score**.

---

## 📊 Dataset

The project uses a Ford used-car dataset containing vehicle specifications and their corresponding selling prices.

### Dataset Features

| Feature | Description |
|---|---|
| `model` | Ford vehicle model |
| `year` | Manufacturing year |
| `price` | Vehicle selling price — target variable |
| `transmission` | Type of transmission |
| `mileage` | Vehicle mileage |
| `fuelType` | Type of fuel used |
| `tax` | Vehicle tax |
| `mpg` | Miles per gallon |
| `engineSize` | Engine size |

### Dataset Size

- **Rows:** 17,966
- **Features:** 9
- **Target Variable:** `price`

---

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify relationships between vehicle characteristics and price.

### EDA Includes

- Dataset preview
- Dataset shape
- Data information
- Descriptive statistics
- Missing-value analysis
- Price distribution
- Correlation heatmap
- Year vs. price analysis
- Mileage vs. price analysis
- Engine size vs. price analysis
- Transmission vs. price analysis
- Fuel type vs. price analysis
- Car model vs. price analysis

### Missing Values

The dataset was checked for missing values across all features.

```text
No missing values were found.
```

---

## 📈 Feature Analysis

Several visualizations were used to investigate how different vehicle characteristics relate to price.

- **Price Distribution:** A histogram with KDE was used to understand the distribution of vehicle prices.
- **Correlation Analysis:** A correlation heatmap was used to examine relationships between numerical variables.
- **Mileage vs Price:** A scatter plot was used to analyze the relationship between mileage and vehicle price.

### Categorical Features
Box plots were used to compare vehicle prices across:
- Transmission types
- Fuel types
- Vehicle models
- Manufacturing years
- Engine sizes

These visualizations provide a better understanding of which vehicle characteristics may influence price.

---

## 🧹 Data Preprocessing

The target variable was separated from the input features.

```python
X = df.drop(columns=['price'])
y = df['price']
```

The categorical features were:
- `model`
- `transmission`
- `fuelType`

Two different encoding approaches were explored.

### 🔤 Encoding Approach 1 — One-Hot Encoding
Categorical features were converted using Pandas `get_dummies()` with `drop_first=True`.

```python
pd.get_dummies(
    X,
    columns=['model', 'transmission', 'fuelType'],
    drop_first=True
)
```

The resulting Boolean features were then converted into integer values. This approach creates separate binary features for the different categories.

### 🔢 Encoding Approach 2 — Label Encoding
A second experiment was performed using `LabelEncoder` for:
- `model`
- `transmission`
- `fuelType`

The encoded categorical features were then included in the numerical feature matrix. This experiment was used to compare the effect of different categorical encoding strategies on Linear Regression performance.

---

## 📏 Feature Scaling

Numerical features were standardized using `StandardScaler`. For the One-Hot Encoding approach, the following numerical features were scaled:
- `year`
- `mileage`
- `tax`
- `mpg`
- `engineSize`

Feature scaling helps bring numerical variables to a comparable scale before model training.

---

## ✂️ Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)
```

### Split Configuration
- **Test Size:** 33%
- **Random State:** 42
- **Test set records:** 5,929

---

## 🤖 Machine Learning Model: Linear Regression

The primary Machine Learning algorithm used in this project is Linear Regression. It was selected as a baseline regression algorithm for predicting continuous vehicle prices.

### 🧪 Model Experimentation

Two Linear Regression models were trained using different categorical encoding strategies.

**Model 1 — One-Hot Encoding**
`One-Hot Encoding` → `Feature Scaling` → `Train/Test Split` → `Linear Regression` → `Evaluation`

**Model 2 — Label Encoding**
`Label Encoding` → `Feature Scaling` → `Train/Test Split` → `Linear Regression` → `Evaluation`

---

## 📊 Model Performance

### Model 1 — One-Hot Encoding
| Metric | Score |
|---|---|
| R² Score | 0.8397 |
| Adjusted R² | 0.8387 |

### Model 2 — Label Encoding
| Metric | Score |
|---|---|
| R² Score | 0.7310 |
| Adjusted R² | 0.7307 |

### Comparison
The One-Hot Encoding approach performed considerably better than the Label Encoding approach for this Linear Regression task.

- **One-Hot Encoding:** R² = 0.8397
- **Label Encoding:** R² = 0.7310

This experiment demonstrates how the choice of categorical encoding can significantly affect Machine Learning model performance.

---

## 🏆 Final Model

The Linear Regression model using One-Hot Encoding produced the best result among the approaches tested in the notebook.

### Final Result
- **R² Score:** 0.8397
- **Adjusted R²:** 0.8387

The model explains approximately 84% of the variance in vehicle prices on the test set.

---

## 🧠 Machine Learning Workflow

`Raw Dataset` ↓
`Dataset Exploration` ↓
`Data Information & Statistics` ↓
`Missing Value Analysis` ↓
`Exploratory Data Analysis` ↓
`Feature/Target Separation` ↓
`Categorical Encoding` ↓
`Feature Scaling` ↓
`Train/Test Split` ↓
`Linear Regression` ↓
`Prediction` ↓
`R² & Adjusted R² Evaluation` ↓
`Encoding Strategy Comparison`

---

## 🛠️ Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

**Scikit-learn Components:**
- `train_test_split`
- `LinearRegression`
- `LabelEncoder`
- `StandardScaler`
- `r2_score`

---

## 📂 Project Structure

```text
02-Car-Price-Prediction/
│
├── Car_price_pred_linear.ipynb
├── ford.csv
└── README.md
```

---



## 💡 Key Learnings

Through this project, I practiced:
- Understanding a real-world regression dataset
- Exploratory Data Analysis
- Descriptive statistics
- Missing-value analysis
- Correlation analysis
- Categorical feature encoding (One-Hot Encoding, Label Encoding)
- Feature scaling
- Train-test splitting
- Linear Regression & Model prediction
- R² Score & Adjusted R²
- Comparing preprocessing strategies
- Understanding the impact of categorical encoding on model performance

---

## 🚀 Future Improvements

The project can be further improved by:
- Comparing multiple regression algorithms
- Trying Ridge and Lasso Regression
- Implementing cross-validation
- Performing hyperparameter tuning
- Adding MAE and RMSE evaluation
- Performing residual/error analysis
- Building a proper Scikit-learn preprocessing pipeline
- Comparing tree-based models such as Random Forest and Gradient Boosting
- Deploying the final model using FastAPI
- Containerizing the prediction service using Docker

---
**👨‍💻 Author:** 
Aditya Kumar Singh
