# Import necessary modules
import numpy as np
import pandas as pd
import pymc3 as pm
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
    with pm.Model() as model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10, shape=np.shape(X)[1])
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value of outcome
        mu = alpha + pm.math.dot(beta, X.T)

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)

        # Inference
        trace = pm.sample(2000, tune=1000)

    # Return the values
    return model, trace

def predict(X, y, features):
    # Get model and model trace
    model, trace = train_model(X, y)
    # Predict the value
    ppc = pm.sample_posterior_predictive(trace, model=model)
    prediction = ppc['y_obs'].mean(axis=0)

    return prediction

