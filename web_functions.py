git clone https://github.com/pgmpy/pgmpy
cd pgmpy/
pip install -r requirements.txt
python setup.py install

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianNetwork
import streamlit as st

@st.cache()
def load_data():
    """This function returns the preprocessed data"""
    df = pd.read_csv('Stress.csv')
    df.rename(columns={"t": "bt"}, inplace=True)
    X = df[["sr", "rr", "bt", "lm", "bo", "rem", "sh", "hr"]]
    y = df['sl']
    return df, X, y

@st.cache()
def train_model(X, y):
    """This function trains the Bayesian Network model and returns the model and its structure"""
    model = BayesianNetwork.from_samples(X, estimator=BayesianEstimator)
    model.fit(X, y)
    return model

def predict(model, features):
    prediction = model.predict(features)
    return prediction
