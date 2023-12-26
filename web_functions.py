"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator  # Corrected import

@st.cache()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('Stress.csv')

    # Rename the column names in the DataFrame.
    df.rename(columns={"t": "bt"}, inplace=True)

    # Perform feature and target split
    X = df[["sr", "rr", "bt", "lm", "bo", "rem", "sh", "hr"]]
    y = df['sl']

    return df, X, y

@st.cache()
def train_model(X, y):
    """This function trains the model and returns the model and model score"""
    # Create the model
    model = BayesianModel()

    # Estimate the parameters using BayesianEstimator
    estimator = BayesianEstimator(model, X)  # Corrected estimator
    model.fit(data=X, estimator=estimator)  # Pass data as a keyword argument

    # Get the model score
    score = model.score(X)

    # Return the values
    return model, score

@st.cache()
def predict(X, y, features):
    model, score = train_model(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
