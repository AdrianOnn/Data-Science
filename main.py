# Importing the necessary Python modules.
import streamlit as st
import pandas as pd
import numpy as np
from pomegranate import BayesianNetwork, DiscreteDistribution, State

# Import necessary functions from web_functions
from web_functions import load_data

# Import pages
from Tabs import home, data, predict_page, visualise

# Configure the app
st.set_page_config(
    page_title='Student Stress Detector',
    page_icon='heavy_exclamation_mark',
    layout='wide',
    initial_sidebar_state='auto'
)

# Dictionary for pages
Tabs = {
    "Home": home,
    "Data Info": data,
    "Prediction": predict_page,
    "Visualisation": visualise
}

# Create a sidebar
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Loading the dataset.
df, X, y = load_data()

# Train model function using Pomegranate
def train_model(X, y):
    model = BayesianNetwork.from_samples(X.values, state_names=X.columns)
    model.fit(X.values, y.values)
    return model

# Prediction function using Pomegranate
def predict(model, features):
    prediction = model.predict(features)
    return prediction

# Call the app function of the selected page to run
if page == "Prediction":
    trained_model = train_model(X, y)
    Tabs[page].app(df, trained_model, predict)
elif page == "Visualisation":
    Tabs[page].app(df)
elif page == "Data Info":
    Tabs[page].app(df)
else:
    Tabs[page].app()
