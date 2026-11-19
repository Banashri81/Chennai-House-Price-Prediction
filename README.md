# Chennai House Price Prediction

A Machine Learning based web application that predicts residential property prices across different localities in Chennai using property features such as area, number of bedrooms, bathrooms, building age, and property type.

## Project Overview

The project uses supervised machine learning algorithms to estimate house prices. It integrates a Flask web application with MongoDB Atlas to store prediction records and display an interactive dashboard.

## Problem Statement

House prices vary across localities based on property size, facilities, building age, and other factors. This project develops a data-driven system to estimate property prices and help users understand property value patterns across Chennai.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- MongoDB Atlas
- HTML and CSS
- JavaScript
- Chart.js
- Jupyter Notebook

## Dataset

The project uses a Chennai house price dataset containing property information such as:

- Locality
- Interior square footage
- Distance from main road
- Number of bedrooms
- Number of bathrooms
- Parking facility
- Building type
- Utility availability
- Street type
- Sale price

## Project Workflow

1. Dataset collection and inspection.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Feature engineering.
5. Machine learning model training.
6. Model comparison.
7. Hyperparameter tuning.
8. Model evaluation.
9. Flask web application development.
10. MongoDB integration.
11. Interactive dashboard development.

## Machine Learning Models

The following algorithms were evaluated:

- Linear Regression
- Support Vector Regression
- Random Forest Regression
- Gradient Boosting Regression

The tuned Random Forest model achieved the best performance among the evaluated models.

## Model Performance

Test dataset results:

| Metric | Score |
|---|---:|
| R² Score | 0.9431 |
| MAE | ₹703,431.68 |
| RMSE | ₹864,727.02 |
| MAPE | 7.71% |

## Application Features

### 1. House Price Prediction
Users enter property details and receive an estimated house price.

### 2. Input Validation
The application checks numerical values and category inputs before prediction.

### 3. MongoDB Integration
Prediction records are stored in MongoDB Atlas.

### 4. Prediction History
Users view previously generated predictions.

### 5. Interactive Dashboard
The dashboard displays:
- Total predictions
- Average predicted price
- Highest predicted price
- Lowest predicted price
- Predictions by locality
- Property type distribution
- Average price by locality

### 6. Model Performance
The dashboard displays the trained model's evaluation metrics.

## Project Structure

```text
Chennai-House-Price-Prediction/
│
├── app.py
├── database.py
├── preprocessing.py
├── train_models.py
├── model_tuning.py
├── feature_importance.py
├── requirements.txt
│
├── data/
│   ├── Chennai houseing sale.csv
│   ├── cleaned_chennai_house_price.csv
│   └── standardized_chennai_house_price.csv
│
├── models/
│   ├── tuned_random_forest_model.pkl
│   ├── model_metrics.json
│   └── tuned_model_metrics.json
│
├── notebooks/
│   └── model_training.ipynb
│
├── results/
│   ├── eda/
│   ├── model_evaluation/
│   └── model_interpretation/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
│
└── static/
    └── style.css
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Banashri81/Chennai-House-Price-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Chennai-House-Price-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure MongoDB

Create a `.env` file in the project root directory.

Add your MongoDB Atlas connection string:

```text
MONGO_URI=your_mongodb_connection_string
```

Never upload your `.env` file or database credentials to GitHub.

### 6. Run the application

```bash
python app.py
```

### 7. Open the application

Visit:

```text
http://127.0.0.1:5000/
```

## Future Improvements

- Add more Chennai localities.
- Integrate real-time property listings.
- Improve model performance with additional property features.
- Add user authentication.
- Deploy the application online.

## Author

Banashri

GitHub: https://github.com/Banashri81

## Project Status

Completed core machine learning pipeline, Flask application, MongoDB integration, prediction history, and interactive dashboard.
