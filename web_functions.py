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
def train_model(X, y):
    """Trains a Bayesian Network model."""

    # Define the network structure (adjust based on domain knowledge)
    model = BayesianModel([("sr", "sl"), ("rr", "sl"), ("bt", "sl"), ("lm", "sl"),
                            ("bo", "sl"), ("rem", "sl"), ("sh", "sl"), ("hr", "sl")])

    # Learn model parameters from data
    model.fit(X, estimator=BayesianEstimator, prior_type="BDeu")

    return model

def predict(model, features):
    """Predicts using the trained Bayesian Network model."""

    prediction = model.predict(features)
    return prediction
