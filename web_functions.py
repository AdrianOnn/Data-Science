import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import streamlit as st

@st.cache()
def load_data():
    """Loads and preprocesses the data."""

    df = pd.read_csv('Stress.csv')
    df.rename(columns={"t": "bt"}, inplace=True)

    X = df[["sr", "rr", "bt", "lm", "bo", "rem", "sh", "hr"]]
    y = df['sl']

    return df, X, y

@st.cache()
def train_model(df):
    model = BayesianModel([('sr', 'sl'), ('rr', 'sl'), ('bt', 'sl'), ('lm', 'sl'),
                           ('bo', 'sl'), ('rem', 'sl'), ('sh', 'sl'), ('hr', 'sl')])
    model.fit(data=df, estimator=BayesianEstimator, prior_type="BDeu")
    return model

def predict(model, features):
    prediction = model.predict_probability(features)
    predicted_stress_level = np.argmax(prediction)
    return predicted_stress_level, prediction
