# Machine Learning for Malaria Prediction Using Clinical and Laboratory Data. Streamlit-app 🩺🦟

## Project Overview

This project focuses on employing machine learning techniques to predict the severity of malaria. Utilizing a rich dataset comprising clinical symptoms, laboratory results, and microscopy, we aim to build a predictive model that can distinguish between non-malaria infections and various levels of malaria severity.

### Dataset Description

The dataset contains detailed clinical and laboratory information from patients suspected to have malaria. The features range from demographic data to intricate laboratory results essential for determining malaria infection.

**Structure**: The dataset is organized such that each instance corresponds to a patient case, with attributes covering basic information like consent and location to clinical details like fever symptoms and suspected organisms. Laboratory findings are extensive, including blood counts and other relevant indicators that might point to the infection's severity.

**Target Variable**: The pivotal element of our dataset is the 'Clinical_Diagnosis' column, which is our target variable. It classifies each case into 'Non-malaria Infection', 'Uncomplicated Malaria', or 'Severe Malaria', and is the outcome we attempt to predict using machine learning models.

### Objective

Our goal is to devise a machine learning model that can predict the severity of malaria with high accuracy, aiding prompt and precise treatment interventions.

### Approach

To reach our objective, we will:
- Perform an exploratory data analysis (EDA) to uncover the underlying structure of the data, distributions of features, and identify any discernible patterns or anomalies.
- Execute data preprocessing to make the dataset conducive to machine learning algorithms. This includes data cleaning, normalization, handling missing data, and selecting significant features.
- Train a variety of machine learning models and evaluate their performance to choose the most efficient one for malaria severity classification.
- Apply appropriate evaluation metrics to validate the accuracy and reliability of our models, ensuring they are robust enough for practical application.

In the end, we anticipate having a sophisticated tool powered by machine learning that healthcare professionals can leverage to improve the diagnosis and treatment of malaria, potentially saving lives and enhancing health outcomes in malaria-prone areas.
1. [Data Overview](#data-overview)
2. [Importing Libraries](#importing-libraries)
3. [Data Cleaning & Preprocessing](#data-cleaning-and-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Univariate Analysis](#univariate-analysis)
    - [Bivariate Analysis](#bivariate-analysis)
    - [Multivariate Analysis](#multivariate-analysis)
5. [Data Encoding](#data-encoding)
6. [Data Scaling](#data-scaling)
7. [Data Modeling](#data-modeling)
8. [Model Evaluation](#model-evaluation)
9. [Pipeline](#pipeline)
10. [Deployment](#deployment)
