# Life Expectancy Prediction using Health and Economic Factors

## Overview

This project aims to predict the life expectancy of countries based on a variety of health, economic, and demographic factors. The dataset used in this project is sourced from Kaggle and contains information on health and economic indicators from various countries. The main goal is to understand the factors influencing life expectancy and create a machine learning model that can accurately predict life expectancy for different countries.

## Dataset Description

The dataset is present in the `data` folder. Please ensure you have downloaded the dataset from Kaggle and saved it in the appropriate location. 

The dataset contains the following columns:

- **Country**: The name of the country.
- **Year**: The year of data collection.
- **Status**: Whether the country is classified as "Developed" or "Developing".
- **Life expectancy**: The life expectancy of individuals in that country (target variable).
- **Adult Mortality**: Death rate among adults aged 15-59.
- **Infant deaths**: Number of infant deaths per 1,000 live births.
- **Alcohol**: Average alcohol consumption per person.
- **Percentage Expenditure**: Percentage of GDP spent on health.
- **Hepatitis B**: Percentage of people vaccinated against Hepatitis B.
- **Measles**: Number of measles cases per 1,000 children.
- **BMI**: Average Body Mass Index (BMI) of the population.
- **Under-five deaths**: Deaths of children under five years old per 1,000 live births.
- **Polio**: Percentage of the population vaccinated against polio.
- **Total expenditure**: Total amount spent on health per person.
- **Diphtheria**: Percentage of the population vaccinated against diphtheria.
- **HIV/AIDS**: Percentage of the population affected by HIV/AIDS.
- **GDP**: Gross Domestic Product per person.
- **Population**: The total population of the country.
- **Thinness (1-19 years)**: Percentage of underweight people aged 1-19.
- **Thinness (5-9 years)**: Percentage of underweight children aged 5-9.
- **Income composition of resources (ICOR)**: A measure of how efficiently a country uses its resources.
- **Schooling**: Average number of years of schooling for the population.

## Objective

The goal of this project is to build a machine learning model that can predict life expectancy based on these features. The key factors influencing life expectancy will be analyzed, and the model will be evaluated for its predictive accuracy.

## Data Preprocessing

Before building the model, the dataset undergoes several preprocessing steps:
1. **Handling missing values**: Missing values are filled with the column mean.
2. **Encoding categorical variables**: One-hot encoding is used to convert categorical columns like `Status` into numerical data.
3. **Feature Scaling**: Data is standardized to ensure all features are on a similar scale.

## Models Evaluated

We evaluated the following regression models to predict life expectancy:

1. **RandomForestRegressor**: An ensemble method using multiple decision trees to improve accuracy.
2. **XGBRegressor**: A gradient boosting model that optimizes for speed and accuracy.
3. **RidgeCV**: Ridge regression with cross-validation to prevent overfitting.
4. **LinearRegression**: A simple linear regression model.
5. **Ridge**: A linear regression model with regularization to prevent overfitting.
6. **GradientBoostingRegressor**: A boosting model that builds trees sequentially to improve performance.

### R² Score Evaluation

Each model was evaluated using the R² (coefficient of determination) score, which indicates the proportion of the variance in life expectancy that can be explained by the model. Here are the results:

| Model                       | R² Score   |
|-----------------------------|------------|
| RandomForestRegressor        | 0.969662   |
| XGBRegressor                 | 0.967337   |
| RidgeCV                      | 0.962525   |
| LinearRegression             | 0.961761   |
| Ridge                        | 0.954758   |
| GradientBoostingRegressor    | 0.952952   |

The **RandomForestRegressor** model achieved the highest R² score of **0.9697**, making it the best model for predicting life expectancy in this dataset.

## Feature Importance

The RandomForestRegressor model also allows us to analyze the importance of each feature in predicting life expectancy. Key findings include:

- **HIV/AIDS**: Strong negative impact on life expectancy.
- **Adult Mortality**: Strong negative correlation with life expectancy.
- **Schooling**: Positive correlation with life expectancy.
- **BMI**, **Thinness (5-9 years)**, **Alcohol**: Moderate impact on life expectancy.

## Insights and Conclusions

From the analysis, we found the following key takeaways:
- **Healthcare and Education**: Countries with higher healthcare expenditure and better education systems tend to have higher life expectancies.
- **Economic Development**: Countries with higher GDP tend to have better life expectancy.
- **Health Factors**: Variables like adult mortality, infant deaths, and HIV/AIDS have a significant negative impact on life expectancy.
- **Feature Engineering**: Features like HIV/AIDS, schooling, and adult mortality were found to be among the most important predictors of life expectancy.

## Future Work

There are several avenues for future work to improve this model:
1. **Data Cleaning**: Addressing data anomalies, such as negative life expectancy values, and improving data quality.
2. **More Features**: Including additional features, such as access to healthcare services or regional factors, could enhance model accuracy.
3. **Model Optimization**: Fine-tuning the models, such as adjusting hyperparameters for XGBoost or Random Forest, could lead to even better performance.

## Requirements

To run this project locally, you will need the following Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
