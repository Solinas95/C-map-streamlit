#!/usr/bin/env python
# coding: utf-8

# In[17]:


import zipfile
import io
import pandas as pd
import streamlit as st

# Define the available configurations
configurations = ["Config 1", "Config 2", "Config 3", "Config 4"]

# Define a function to load a dataframe from a text file in the ZIP archive
def load_df_from_zip(zip_file, file_name):
    with zip_file.open(file_name, "r") as file:
        file_content = file.read().decode("ascii")
        df_name = file_name.split(".")[0].split("_")[0]  # Extract the dataframe name from the file name
        df = pd.read_csv(io.StringIO(file_content), sep=" ", header=None)
        return df_name, df

# Streamlit app
st.title("Data Loading Tool")

# Allow the user to select a configuration
selected_config = st.selectbox("Select a configuration", options=configurations)

# Open the ZIP file
with zipfile.ZipFile("C-maps data.zip", "r") as zip_file:
    # Load the dataframes for the selected configuration
    dataframes = {}
    for file_name in zip_file.namelist():
        if file_name.endswith(".txt") and selected_config in file_name:
            df_name, df = load_df_from_zip(zip_file, file_name)
            dataframes[df_name] = df

    # Show the loaded dataframes
    st.write("Loaded dataframes:")
    for df_name, df in dataframes.items():
        st.write(f"{df_name}:")
        st.write(df.head())

