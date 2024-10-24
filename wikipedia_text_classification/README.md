# Text Classification using Logistic Regression on Wikipedia Biographies

## Project Overview

This project demonstrates how to build a text classifier to predict a person's nationality based on their Wikipedia biography. We employ various text processing techniques and build a logistic regression model to classify the text data accurately. The project leverages the scikit-learn library for model training and evaluation, focusing on processing and cleaning the data to enhance model performance.

## Table of Contents

1. [Project Description](#project-description)
2. [Data Import and Preparation](#data-import-and-preparation)
3. [Tokenization Methods](#tokenization-methods)
4. [Train-Test Split](#train-test-split)
5. [Baseline and Logistic Regression Models](#baseline-and-logistic-regression-models)
6. [Model Evaluation](#model-evaluation)
7. [Feature Analysis](#feature-analysis)
8. [Setup Instructions](#setup-instructions)
9. [Key Results](#key-results)
10. [Contact Information](#contact-information)

---

## Project Description

This project aims to build a text classifier using logistic regression to predict the nationality of individuals based on their Wikipedia biographies. The classification process involves data cleaning, feature extraction using TF-IDF, and model evaluation using macro-averaged F1 scores. Additionally, we explore different tokenization methods and establish baseline models for performance comparison.

## Data Import and Preparation

- **Data Source:** The dataset is imported from `../../assets/bio_nationality.tsv.gz`.
- **Data Cleaning:** Nationality labels are standardized to remove redundancy and ensure consistency.
- **Filtering:** Labels occurring less than 500 times are filtered out to maintain sufficient training examples per class.
- **Train-Test Split:** Data is split into training, development, and test sets with proportions of 80%, 10%, and 10%, respectively.

## Tokenization Methods

Tokenization is performed using three methods:
- **Simple Split:** Splits text by whitespace.
- **sklearn Split:** Uses regular expressions similar to `TfidfVectorizer` in scikit-learn.
- **nltk Split:** Utilizes `word_tokenize` from the nltk library.

## Train-Test Split

The dataset is split into training, development, and test sets. The sizes are:
- Training Set: 41,544 entries
- Development Set: 5,193 entries
- Test Set: 5,194 entries

## Baseline and Logistic Regression Models

- **Baseline Models:** Implemented using `DummyClassifier` with 'uniform' and 'stratified' strategies.
- **Logistic Regression:** Trained on TF-IDF features extracted from the biographies.

## Model Evaluation

Models are evaluated based on macro-averaged and micro-averaged F1 scores. Confusion matrices and accuracy per class are also analyzed to understand model performance and common confusions.

## Feature Analysis

- **Most Informative Features:** The logistic regression coefficients are analyzed to identify the most informative words for each nationality.
- **Confusion Matrix:** A heatmap of the confusion matrix is plotted to visualize model performance and errors.

![image](https://github.com/user-attachments/assets/352a833d-f080-4ef8-ac2f-e73d00c0d05a)


## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/akamal341/text-classification-wikipedia.git
    cd text-classification-wikipedia
    ```

2. **Install Dependencies**:
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn nltk tqdm
    ```

3. **Run the Jupyter Notebook**:
    Open and execute the provided Jupyter Notebook `text_classification_project.ipynb` to walk through the data analysis, feature engineering, model training, and evaluation.

## Key Results

1. **Baseline Models:** Provided points of comparison with very low F1 scores (uniform: 0.041, stratified: 0.052).
2. **Logistic Regression Model:** Achieved a macro-averaged F1 score of 0.762 on the development set.
3. **Feature Analysis:** Identified key words indicative of different nationalities.
4. **Confusion Matrix:** Visualized model errors and insights for performance improvements.

## Contact Information

For any questions or further information, please contact:
- **Name:** Asad Kamal
- **Email:** aakamal {/@/} umich {/dot/} edu
- **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/asadakamal)
