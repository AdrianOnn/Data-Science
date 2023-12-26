import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import streamlit as st

@st.cache()
def load_data():
    # Print available columns to check for typos
    with open('Stress.csv', 'r') as f:
        cols = pd.read_csv(f, nrows=1).columns
        print("Available columns:", cols)

    # Load only necessary columns and convert data types
    df = pd.read_csv('Stress.csv', usecols=['sr', 'rr', 'bt', 'lm', 'bo', 'rem', 'sh', 'hr', 'sl'])
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['sl'] = df['sl'].astype('category')

    # Feature and target split
    X = df.drop('sl', axis=1)  # Avoid creating a copy using iloc
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
