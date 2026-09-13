# Emotion Detection ML

An NLP-based machine learning mini project that classifies text into different emotion categories using text preprocessing, feature extraction, and traditional machine learning algorithms.

## 📌 Project Overview

**Emotion Detection ML** is a Natural Language Processing (NLP) mini project designed to classify text according to the emotion expressed in it.

The project follows a complete machine learning workflow, including data exploration, text preprocessing, label encoding, feature extraction, model training, and performance comparison.

The project compares **Bag-of-Words with Multinomial Naive Bayes**, **TF-IDF with Multinomial Naive Bayes**, and **TF-IDF with Logistic Regression**.

## 🎯 Objective

The main objective of this project is to build and compare different machine learning approaches for text-based emotion classification and identify the best-performing model based on test accuracy.

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
├── Emotion_Detection_ML.ipynb
├── train.txt
└── README.md
```

## 📊 Dataset

The dataset contains **16,000 text samples** belonging to **6 different emotion classes**.

### Dataset Details

* Total samples: **16,000**
* Unique text samples: **15,969**
* Emotion classes: **6**
* Missing values: **None**

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

using `random_state=42`.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
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
   ├── CountVectorizer
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

The text data is cleaned before being used for machine learning.

The preprocessing steps include:

* Converting text to lowercase
* Removing punctuation
* Removing numbers
* Removing non-ASCII characters and emojis
* Removing English stopwords using NLTK

### Example

**Before:**

```text
i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake
```

**After:**

```text
go feeling hopeless damned hopeful around someone cares awake
```

## 🤖 Machine Learning Models

### 1. CountVectorizer + Multinomial Naive Bayes

The text is converted into a Bag-of-Words representation using `CountVectorizer` and classified using `MultinomialNB`.

**Test Accuracy: 76.8125%**

### 2. TF-IDF + Multinomial Naive Bayes

TF-IDF is used to transform the text into numerical features, followed by Multinomial Naive Bayes classification.

**Test Accuracy: 66.09375%**

### 3. TF-IDF + Logistic Regression

TF-IDF features are used with Logistic Regression for multiclass emotion classification.

**Test Accuracy: 86.28125%**

## 📈 Model Comparison

| Model                   | Feature Extraction | Test Accuracy |
| ----------------------- | ------------------ | ------------: |
| Multinomial Naive Bayes | CountVectorizer    |      76.8125% |
| Multinomial Naive Bayes | TF-IDF             |     66.09375% |
| Logistic Regression     | TF-IDF             | **86.28125%** |

## 🏆 Best Model

The best-performing model is:

**TF-IDF + Logistic Regression**

with a test accuracy of **86.28125%**.

This model achieved the highest accuracy among the three approaches evaluated in the project.

## 📚 Key Learning Outcomes

This project provides practical experience with:

* Natural Language Processing
* Text preprocessing
* Stopword removal
* Label encoding
* Bag-of-Words representation
* TF-IDF feature extraction
* Multinomial Naive Bayes
* Logistic Regression
* Train-test splitting
* Model evaluation
* Comparing machine learning approaches

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/connectwithaadi/Emotion-Detection-ML.git
```

### 2. Navigate to the project directory

```bash
cd Emotion-Detection-ML
```

### 3. Install the required libraries

```bash
pip install numpy pandas nltk scikit-learn matplotlib seaborn
```

### 4. Open the notebook

Open:

```text
Emotion_Detection_ML.ipynb
```

The notebook can be run using **Jupyter Notebook** or **Google Colab**.

> **Note:** The notebook currently uses the dataset path `/content/train.txt`. If you run it locally, update the dataset path according to your local file location.

## 🔮 Future Improvements

The project can be further improved by:

* Hyperparameter tuning
* Using additional evaluation metrics such as Precision, Recall, and F1-score
* Experimenting with different NLP preprocessing techniques
* Testing additional machine learning algorithms
* Using larger or more diverse datasets
* Exploring advanced NLP models

## 👨‍💻 Author

**Aditya Kumar Singh**


---

