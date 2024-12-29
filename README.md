# 📊 Machine Learning Sales Prediction

Welcome to the **Machine Learning Sales Prediction** project! This repository provides a collection of machine learning algorithms to predict sales using different features like store details, promotions, and other factors.

---

## 📁 Table of Contents

1. [Project Overview](#project-overview)
2. [Datasets](#datasets)
3. [Algorithms Used](#algorithms-used)
4. [Features](#features)
5. [Project Structure](#project-structure)
6. [Installation Instructions](#installation-instructions)
7. [Evaluation Metrics](#evaluation-metrics)
8. [How to Run](#how-to-run)
9. [License](#license)

---

## 📝 Project Overview

This project is designed to predict sales for various stores using historical data. Machine learning models are trained on different features such as store location, promotions, and customer count. Multiple algorithms are tested to provide the best possible sales predictions.

---

## 📊 Datasets

This project uses three different datasets that contain crucial information for predicting sales:

### 1. **Dataset 1: Sales Data** 📅
   - Contains historical sales data for stores.
   - Includes columns like `Date`, `Store`, `DayOfWeek`, `Promo`, `Sales`, etc.
   - This dataset is used to train and predict future sales.

### 2. **Dataset 2: Store Information** 🏬
   - Contains store-specific data, including location and features like `CompetitionDistance` and `Promo2`.
   - These features are critical to understanding sales patterns and predicting future sales.

### 3. **Dataset 3: Promo Data** 💥
   - This dataset contains promotional data for stores that can help in understanding the impact of discounts, special offers, and seasonal promotions on sales.

---

## 🔧 Algorithms Used

The following machine learning algorithms have been implemented for sales prediction:

1. **Gradient Boosting Regressor** 💡
   - A powerful ensemble model that combines weak learners to produce robust predictions.

2. **K-Nearest Neighbors (KNN)** 🔍
   - A simple yet effective algorithm that classifies data points based on the closest neighbors.

3. **Decision Tree Regressor** 🌳
   - A non-linear model that splits the dataset into branches based on feature values.

4. **Random Forest Regressor** 🌲
   - An ensemble method that uses multiple decision trees to improve prediction accuracy.

5. **Ridge Regression** 📉
   - A linear model that applies L2 regularization to prevent overfitting.

6. **Linear Regression** ➗
   - The simplest form of regression that predicts sales based on a linear relationship with features.

7. **RANSAC Regressor** 🔧
   - A robust method that fits the model by ignoring outliers in the dataset.

8. **ElasticNet Regression** ⚙️
   - Combines the benefits of both Lasso and Ridge Regression.

9. **XGBoost Regressor** 🚀
   - A highly efficient and scalable implementation of gradient boosting.

10. **LightGBM Regressor** 💨
    - A faster, more efficient gradient boosting framework optimized for large datasets.

---

## 📈 Features

The dataset contains several features that help in making accurate sales predictions:

- **Store** 🏬
- **DayOfWeek** 📅
- **Customers** 👥
- **Open** 🏪
- **Promo** 💥
- **SchoolHoliday** 🎓
- **CompetitionDistance** 🏙️
- **CompetitionOpenSinceMonth** 🗓️
- **Promo2SinceYear** 🕐
- **ItemPrice** 💸

---

## 🛠️ Project Structure
Machine-Learning-Sales-Prediction/ │ ├── data/ │ ├── dataset1.csv # Sales data │ ├── dataset2.csv # Store information │ ├── dataset3.csv # Promotional data │ ├── models/ │ ├── gradient_boosting_model.pkl # Trained Gradient Boosting model │ ├── random_forest_model.pkl # Trained Random Forest model │ ├── xgboost_model.pkl # Trained XGBoost model │ ├── scripts/ │ ├── data_preprocessing.py # Data cleaning and preprocessing │ ├── model_training.py # Model training script │ ├── model_evaluation.py # Evaluation of models │ ├── README.md # Project documentation ├── requirements.txt # List of dependencies └── .gitignore # Git ignore file


## ⚙️ Installation Instructions

To get started with the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Machine-Learning-Sales-Prediction.git
    cd Machine-Learning-Sales-Prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 Evaluation Metrics

The performance of the models is evaluated using the following metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of the MSE, provides a more interpretable error value.
- **R-Squared (R²)**: Indicates how well the model explains the variance of the data.
- **Mean Absolute Percentage Error (MAPE)**: Measures the percentage error between predicted and actual values.

---

## 🏃‍♂️ How to Run

1. Preprocess the data:
    ```bash
    python scripts/data_preprocessing.py
    ```

2. Train models:
    ```bash
    python scripts/model_training.py
    ```

3. Evaluate models:
    ```bash
    python scripts/model_evaluation.py
    ```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



