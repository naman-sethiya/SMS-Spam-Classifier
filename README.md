# SMS Spam Classifier

## Overview

This project builds an SMS spam classifier using machine learning. It involves data cleaning, exploratory data analysis (EDA), feature engineering, model building, and evaluation to classify SMS messages as spam or ham.

## Key Steps

1. **Data Cleaning**: Preprocess the text data by removing unwanted characters and handling missing values.
  
2. **Exploratory Data Analysis (EDA)**: Analyze the distribution of spam and ham messages, and text characteristics like length and structure.

3. **Feature Engineering**: Convert text into numerical features using TF-IDF vectorization.

4. **Model Building**: Train a Multinomial Naive Bayes model with TF-IDF features.

5. **Model Evaluation**: Evaluate the model's performance, achieving a precision of 99.1% and an accuracy of 98.1%.

## Requirements

- Python 3.x
- NLTK
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone https://github.com/naman-sethiya/SMS-Spam-Classifier.git
cd SMS-Spam-Classifier
pip install -r requirements.txt
