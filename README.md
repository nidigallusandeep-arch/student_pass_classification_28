# 🎓 Student Academic Performance Prediction Using Machine Learning

## 🌐 Live Demo

**Try the application here:**  
[Student Pass/Fail Prediction App](https://studentpassclassification28-bgqiqhsrdluwlacbrmwhmt.streamlit.app/)

## 📌 Project Overview

The Student Academic Performance Prediction project is a Machine Learning web application built using Python, Scikit-learn, and Streamlit.

It predicts whether a student is likely to **Pass or Fail** based on study hours, attendance, previous scores, course selection, internet access, and study method.

The application provides an interactive interface where users can enter student details and view predictions and model evaluation metrics.

## 🎯 Objectives

- Predict student Pass/Fail outcomes using Logistic Regression.
- Handle missing values in numerical and categorical features.
- Convert categorical features into numerical representations.
- Scale numerical features for model training.
- Optimize model hyperparameters using GridSearchCV.
- Deploy the application using Streamlit Community Cloud.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- OneHotEncoder
- StandardScaler
- SimpleImputer
- GridSearchCV
- Git and GitHub

## 📊 Dataset Description

The project uses a synthetic dataset containing 180 student records and 7 columns.

| Feature | Description |
|---|---|
| `Study_Hours` | Daily study hours |
| `Attendance` | Attendance percentage |
| `Previous_Score` | Previous examination score |
| `Course` | Selected course |
| `Internet_Access` | Internet availability |
| `Study_Method` | Method of studying |
| `Pass` | Target variable: 0 = Fail, 1 = Pass |

**Note:** This dataset is synthetically generated for educational and demonstration purposes.

## ⚙️ Machine Learning Workflow

1. Load the dataset using Pandas.
2. Separate input features and target variable.
3. Split data into training and testing sets.
4. Handle missing numerical values using median imputation.
5. Handle missing categorical values using most-frequent imputation.
6. Apply StandardScaler to numerical features.
7. Apply OneHotEncoder to categorical features.
8. Train a Logistic Regression model.
9. Tune hyperparameters using GridSearchCV.
10. Evaluate the model using classification metrics.
11. Build an interactive Streamlit application.
12. Deploy the application using Streamlit Community Cloud.

## 🤖 Algorithm and Hyperparameter Tuning

**Algorithm:** Logistic Regression

Hyperparameters tuned using GridSearchCV:

- `C`: 0.01, 0.1, 1, 10, 100
- `solver`: `liblinear`, `lbfgs`
- Cross-validation: 5 folds

The best combination is selected using cross-validation accuracy.

## 📈 Evaluation Metrics

The application displays:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-score
- Cross-validation accuracy

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
cd student-performance-prediction
```

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository URL.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 📁 Project Structure

```text
student-performance-prediction/
│
├── app.py
├── student_pass_classification.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## 🌍 Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:**  
https://studentpassclassification28-bgqiqhsrdluwlacbrmwhmt.streamlit.app/

## ⚠️ Disclaimer

This project is intended for learning and demonstration. Predictions are generated from synthetic practice data and should not be used to make real academic decisions. Model performance on this dataset does not establish real-world predictive accuracy.

## 👨‍💻 Author

Developed as a Machine Learning project to practice data preprocessing, categorical encoding, classification, hyperparameter tuning, model evaluation, and Streamlit deployment.

