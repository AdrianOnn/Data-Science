# Importing the necessary Python modules.
import streamlit as st

# Import necessary functions from web_functions
from web_functions import load_data, train_model, predict

# Import pages
from Tabs import home, data, predict_page, visualise

# Configure the app
st.set_page_config(
    page_title='Stress Level Detector',
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

# Call the app function of the selected page to run
if page == "Prediction":
    Tabs[page].app(df, X, y, train_model, predict)
elif page == "Visualisation":
    Tabs[page].app(df)
elif page == "Data Info":
    Tabs[page].app(df)
else:
    Tabs[page].app()
