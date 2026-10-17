import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Student Pass/Fail Prediction", page_icon="🎓")
st.title("🎓 Student Pass/Fail Prediction")
st.write("Predict a student's result using Logistic Regression.")

@st.cache_data
def load_data():
    return pd.read_csv("student_pass_classification.csv")

@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop(columns=["Pass"])
    y = df["Pass"]

    num_cols = ["Study_Hours", "Attendance", "Previous_Score"]
    cat_cols = ["Course", "Internet_Access", "Study_Method"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])
    params = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"]
    }
    grid = GridSearchCV(
        pipeline, params, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    return df, best_model, grid.best_params_, grid.best_score_, accuracy, cm, report

try:
    df, model, best_params, cv_accuracy, test_accuracy, cm, report = train_model()
except FileNotFoundError:
    st.error(
        "CSV file not found. Put 'student_pass_classification.csv' in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)

st.subheader("Enter student details")
with st.form("student_form"):
    c1, c2 = st.columns(2)
    with c1:
        study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        previous_score = st.number_input("Previous score", min_value=0.0, max_value=100.0, value=72.0, step=1.0)
    with c2:
        course = st.selectbox("Course", ["Python", "Data Science", "Web Development"])
        internet = st.selectbox("Internet access", ["Yes", "No"])
        study_method = st.selectbox("Study method", ["Self Study", "Coaching", "Online Course"])
    submitted = st.form_submit_button("Predict result")

if submitted:
    student = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Course": course,
        "Internet_Access": internet,
        "Study_Method": study_method
    }])
    prediction = int(model.predict(student)[0])
    probability = float(model.predict_proba(student)[0, 1])
    if prediction == 1:
        st.success("Prediction: PASS")
    else:
        st.warning("Prediction: FAIL")
    st.metric("Estimated probability of passing", f"{probability:.1%}")
    st.caption("This is a prediction from a synthetic practice dataset, not a real academic assessment.")

st.divider()
st.subheader("Model performance")
m1, m2 = st.columns(2)
m1.metric("Test accuracy", f"{test_accuracy:.1%}")
m2.metric("Best cross-validation accuracy", f"{cv_accuracy:.1%}")
st.write("Best hyperparameters:", best_params)
st.write("Confusion matrix (actual rows, predicted columns):")
st.dataframe(pd.DataFrame(cm, index=["Actual Fail (0)", "Actual Pass (1)"], columns=["Predicted Fail (0)", "Predicted Pass (1)"]))
st.write("Classification report:")
st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
