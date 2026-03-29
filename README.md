# Customer Purchase Prediction

> An end-to-end Machine Learning web application that predicts whether a customer will make a purchase — from raw data to a live, deployed prediction interface.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=flat-square)

---

## Why I Built This

Most ML tutorials stop at model training. I wanted to build the full loop — data in, model trained, predictions served live through a real web interface. This project covers every stage a working ML system goes through: cleaning messy data, building a pipeline, evaluating the model honestly, and deploying it so anyone can use it without touching code.

---

## What It Does

Given customer features (e.g. age, browsing history, session duration), the model predicts the probability that the customer will complete a purchase. The Flask web interface lets users input customer data and receive a real-time prediction instantly.

---

## ML Pipeline

```
Raw Data
   │
   ▼
Data Preprocessing       ← handle missing values, encode categoricals
   │
   ▼
Feature Scaling          ← StandardScaler to normalize input range
   │
   ▼
Model Training           ← Logistic Regression (sklearn)
   │
   ▼
Model Evaluation         ← accuracy, precision, recall, confusion matrix
   │
   ▼
Model Serialization      ← saved with joblib / pickle
   │
   ▼
Flask Web Interface      ← user inputs features → live prediction returned
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| ML Model | Logistic Regression (Scikit-learn) |
| Data Processing | Pandas, NumPy |
| Feature Scaling | StandardScaler (Scikit-learn) |
| Web Framework | Flask |
| Model Serialization | Joblib |

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | _(add your value)_ |
| Precision | _(add your value)_ |
| Recall | _(add your value)_ |

> Update the table above with your actual evaluation results before publishing.

---

## How to Run

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/customer-purchase-prediction.git
cd customer-purchase-prediction

# Install dependencies
pip install -r requirements.txt

# Train the model (generates saved model file)
python train.py

# Run the web app
python app.py

# Open in browser
http://localhost:5000
```

---

## Project Structure

```
customer-purchase-prediction/
├── data/
│   └── customer_data.csv       # Raw dataset
├── notebooks/
│   └── exploration.ipynb       # EDA and model experimentation
├── src/
│   ├── preprocess.py           # Data cleaning and feature engineering
│   ├── train.py                # Model training and evaluation
│   └── model.pkl               # Saved trained model
├── templates/
│   └── index.html              # Flask frontend
├── app.py                      # Flask application entry point
├── requirements.txt
└── README.md
```

---

## Key Concepts Demonstrated

- **End-to-end ML pipeline** — from raw CSV to deployed web prediction
- **Data preprocessing** — handling nulls, encoding, and feature engineering
- **Feature scaling** — why StandardScaler matters for Logistic Regression
- **Model evaluation** — reading beyond accuracy to precision and recall
- **ML deployment** — serializing a model and serving predictions via REST with Flask
