# Stroke-Prediction
## Project Overview 🚀
This project focuses on predicting the likelihood of a stroke based on various health and demographic factors. The goal is to build a reliable machine learning model to assist in early detection and prevention, ultimately contributing to better healthcare outcomes.

## Problem Statement ❗
Stroke is a leading cause of death and disability worldwide. Early prediction can significantly improve patient outcomes by enabling timely medical intervention. This project aims to address the challenge of accurately predicting stroke occurrences using available data.

## Dataset Used 📊
The dataset used for this project is sourced from [Kaggle's Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). It contains information on features such as:
- Age
- Gender
- Hypertension
- Heart disease
- Smoking status
- BMI, and more  
The target variable indicates whether a stroke occurred.

## EDA and Preprocessing 🔍
Key steps in Exploratory Data Analysis (EDA) and preprocessing included:
- **Analyzing feature distributions** to understand data patterns.
- **Handling missing values** (e.g., imputing missing BMI values).
- **Encoding categorical variables** using one-hot encoding.
- **Scaling numerical features** for consistency.
- **Balancing the dataset** using SMOTE to address class imbalance.
- **Outlier detection and removal** to improve model robustness.

## Models Used and Evaluation Metrics 🤖
Several machine learning models were implemented and evaluated:
- **Logistic Regression**
- **Naive Bayes**
- **Decision Tree**
- **KNN**
- **SVM**
Evaluation metrics included:
- Accuracy ✅
- Precision 🎯
- Recall 🔄
- F1-score ⚖️
- AUC-ROC 📈

## Key Results and Takeaways 🏆
- The **Gradient Boosting model** achieved the highest performance with an **Recall of 0.91** and balaced scores in other metrics. 🥇
- **Feature importance analysis** revealed that **age**, **hypertension**, and **heart disease** were the most significant predictors. 📌
- Proper handling of **class imbalance** (using SMOTE) significantly improved model performance. ⚡
- **Hyperparameter tuning** further optimized model accuracy and generalization. 🔧
- Visualizations like **correlation heatmaps** and **pair plots** provided deeper insights into feature relationships. 📊
- The project underscores the importance of **data preprocessing** and **feature engineering** in achieving accurate predictions. 🛠️
- Early detection models like this can play a crucial role in reducing stroke-related mortality and morbidity. ❤️‍🩹
- Future work could explore **deep learning models** and **real-time prediction systems** for further advancements. 🚀