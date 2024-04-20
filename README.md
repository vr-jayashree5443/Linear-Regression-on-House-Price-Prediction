# Linear-Regression-on-House-Price-Prediction

## Introduction
This project aims to predict house prices using machine learning techniques. The dataset used for training the model is the Boston dataset, which contains various features related to housing in different towns.

## Problem Statement
Given the Boston dataset with features related to housing, the goal is to train a machine learning model to predict the prices of houses.

## Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation
Ensure you have scikit-learn installed:
```
pip install scikit-learn
```

## Dataset
The dataset used is the Boston dataset, which can be accessed from the following URL: [Boston Dataset](http://lib.stat.cmu.edu/datasets/boston). This dataset contains various features such as crime rate, proportion of residential land, nitric oxides concentration, etc.

## Exploratory Data Analysis (EDA)
- Univariate, bivariate, and multivariate EDA techniques were applied.
- Descriptive statistics and visualizations were utilized to understand the dataset.
- Features were analyzed for correlations and relationships with the target variable.

## Machine Learning - Linear Regression
- Linear Regression model was chosen for predicting house prices.
- The dataset was split into training and testing sets.
- The model was trained using the training data.
- Predictions were made on the testing data.
- Model performance was evaluated using R squared, Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Future Scope
- Utilizing other regression algorithms such as Random Forest for potentially better accuracy.
- Exploring automated machine learning (AutoML) for selecting the best model.
- Further optimization of hyperparameters for improved model performance.

## klib Library
- klib is a Python library for importing, cleaning, analyzing, and preprocessing data.
- It provides functions for visualizing datasets, data cleaning, and handling missing values.
- Users can explore and utilize klib functions to streamline their data preprocessing tasks.

## Usage of klib Library
- Example functions include `klib.cat_plot()`, `klib.corr_mat()`, `klib.dist_plot()`, etc.
- klib functions can assist in data cleaning, visualization, and preprocessing tasks.
- Users are encouraged to understand how each function works and utilize them accordingly.

## Conclusion
The project successfully demonstrates the prediction of house prices using the Boston dataset and a Linear Regression model. Through thorough EDA and model evaluation, insights into housing prices were gained. Additionally, the utilization of libraries like klib can aid in efficient data preprocessing and analysis. Further improvements and exploration can be done to enhance model accuracy and performance.
