import streamlit as st
from web_functions import load_data, train_model  # Import model training function

# Import pages
from Tabs import home, data, predict, visualise

# Configure the app
st.set_page_config(
    page_title="Student Stress Detector",
    page_icon="heavy_exclamation_mark",
    layout="wide",
    initial_sidebar_state="auto"
)

# Dictionary for pages
Tabs = {
    "Home": home,
    "Data Info": data,
    "Prediction": predict,
    "Visualisation": visualise
}

# Create a sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Load the dataset and train the model
df, X, y = load_data()
model = train_model(X, y)  # Train the Bayesian Network model

# Call the app function of the selected page
if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, X, model)  # Pass the trained model to the pages
elif page == "Data Info":
    Tabs[page].app(df)
else:
    Tabs[page].app()
