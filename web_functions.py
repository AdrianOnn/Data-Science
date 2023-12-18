# Import necessary modules
import numpy as np
import pandas as pd
from pomegranate.bayesian_network import BayesianNetwork
import streamlit as st

@st.cache()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('Stress.csv')

    # Rename the column names in the DataFrame.
    df.rename(columns = {"t": "bt",}, inplace = True)
    
    # Perform feature and target split
    X = df[["sr","rr","bt","lm","bo","rem","sh","hr"]]
    y = df['sl']

    return df, X, y

@st.cache()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # Create the model
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')

    # Return the model
    return model

def predict(X, y, features):
    # Get model
    model = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction
