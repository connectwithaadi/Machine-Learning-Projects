# Emotion Detection ML

An NLP-based machine learning mini project that classifies text into different emotion categories using text preprocessing and traditional machine learning techniques.

## 📌 Project Overview

**Emotion Detection ML** is a Natural Language Processing (NLP) project that analyzes textual data and predicts the emotion associated with the given text.

The project demonstrates a complete basic NLP classification workflow, including:

* Text preprocessing
* Stopword removal
* Label encoding
* Bag-of-Words feature extraction
* TF-IDF feature extraction
* Machine learning model training
* Model comparison using accuracy

## 🎯 Objective

The main objective of this project is to build and compare different machine learning approaches for **text emotion classification** and identify the model that performs best on the given dataset.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook / Google Colab

## 📂 Project Structure

```text
Emotion-Detection-ML/
│
├── Emotion-Detection-ML.ipynb
├── README.md
└── dataset/
    └── test.txt
```

## 🔄 Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
Stopword Removal
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
Feature Extraction
   ├── Bag-of-Words
   └── TF-IDF
   ↓
Model Training
   ├── Multinomial Naive Bayes
   └── Logistic Regression
   ↓
Model Evaluation
   ↓
Accuracy Comparison
```

## 🧹 Text Preprocessing

The text data is cleaned and prepared before training the models.

The preprocessing steps include:

* Converting text to lowercase
* Removing punctuation
* Removing numbers
* Removing non-ASCII characters and emojis
* Removing English stopwords using NLTK

### Example

**Before:**

```text
im updating my blog because i feel shitty
```

**After:**

```text
im updating blog feel shitty
```

## 📊 Dataset

The dataset contains **2,000 text samples** belonging to **6 different emotion classes**.

* Total samples: **2,000**
* Unique text samples: **2,000**
* Number of emotion classes: **6**
* Missing values: **None**

The dataset is split into:

* **80% Training Data**
* **20% Testing Data**

with `random_state=42`.

## 🤖 Machine Learning Models

### 1. CountVectorizer + Multinomial Naive Bayes

Bag-of-Words representation is created using `CountVectorizer`, followed by a Multinomial Naive Bayes classifier.

**Accuracy: 61.50%**

### 2. TF-IDF + Multinomial Naive Bayes

TF-IDF is used for feature extraction and Multinomial Naive Bayes is used for classification.

**Accuracy: 57.50%**

### 3. TF-IDF + Logistic Regression

TF-IDF features are combined with Logistic Regression for multiclass emotion classification.

**Accuracy: 63.75%**

## 📈 Results

| Model                   | Feature Extraction |   Accuracy |
| ----------------------- | ------------------ | ---------: |
| Multinomial Naive Bayes | CountVectorizer    |     61.50% |
| Multinomial Naive Bayes | TF-IDF             |     57.50% |
| Logistic Regression     | TF-IDF             | **63.75%** |

### 🏆 Best Model

The best-performing approach in this project is:

**TF-IDF + Logistic Regression**

with a test accuracy of **63.75%**.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/connectwithaadi/Emotion-Detection-ML.git
```

### 2. Navigate to the project directory

```bash
cd Emotion-Detection-ML
```

### 3. Install dependencies

```bash
pip install numpy pandas nltk scikit-learn matplotlib seaborn
```

### 4. Open the notebook

Open:

```text
Emotion-Detection-ML.ipynb
```

You can run it using **Jupyter Notebook** or **Google Colab**.

> **Note:** If running the notebook in a different environment, update the dataset path according to your local dataset location.

## 📌 Key Learning Outcomes

Through this project, I explored:

* Fundamentals of NLP
* Text cleaning and preprocessing
* Stopword removal
* Label encoding
* Bag-of-Words
* TF-IDF
* Multinomial Naive Bayes
* Logistic Regression
* Train-test splitting
* Comparing machine learning models using accuracy

## 🔮 Future Improvements

The project can be extended in the future by:

* Adding more training data
* Performing hyperparameter tuning
* Using additional NLP techniques
* Evaluating models using precision, recall, and F1-score
* Experimenting with advanced NLP models

## 👨‍💻 Author

**Aditya Kumar Singh**


