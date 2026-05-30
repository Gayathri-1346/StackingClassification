import os
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from sklearn.metrics import accuracy_score


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(BASE_DIR, "data", "heart.csv"),
    os.path.join(BASE_DIR, "heart.csv")
]

for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        break
else:
    raise FileNotFoundError(
        f"heart.csv not found. Checked: {possible_paths}"
    )

os.makedirs("model", exist_ok=True)

MODEL_PATH = "model/stacking_classifier.pkl"

# Training

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA_PATH)

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    categorical_cols = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope"
    ]

    numerical_cols = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_cols
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            )
        ]
    )

    base_models = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        ),
        (
            "gb",
            GradientBoostingClassifier(
                random_state=42
            )
        )
    ]

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression()
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        preds
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline, accuracy


model, accuracy = train_model()

# UI

st.title("Heart Disease Prediction")
st.subheader("Stacking Classifier")

age = st.slider("Age", 20, 80, 40)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

bp = st.number_input(
    "Resting Blood Pressure",
    80,
    220,
    120
)

chol = st.number_input(
    "Cholesterol",
    0,
    700,
    200
)

fasting = st.selectbox(
    "Fasting Blood Sugar > 120",
    [0, 1]
)

ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "LVH", "ST"]
)

maxhr = st.slider(
    "Max Heart Rate",
    60,
    220,
    150
)

angina = st.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Old Peak",
    0.0,
    6.0,
    1.0
)

slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [chest],
    "RestingBP": [bp],
    "Cholesterol": [chol],
    "FastingBS": [fasting],
    "RestingECG": [ecg],
    "MaxHR": [maxhr],
    "ExerciseAngina": [angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [slope]
})

if st.button("Predict"):

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")

st.divider()

st.metric(
    "Model Accuracy",
    f"{accuracy:.2%}"
)

st.dataframe(input_df)
