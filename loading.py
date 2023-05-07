#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import streamlit as st
import pandas as pd

# Set title and description
st.title("Data Uploader")
st.write("Please upload your train and test data in CSV format.")

# Function to load data from uploaded file
def load_data(data):
    return pd.read_csv(data)

# Upload train data
train_data_file = st.file_uploader("Upload Train Data (CSV)", type="csv")
if train_data_file is not None:
    train_data = load_data(train_data_file)
    st.write("Train Data:")
    st.write(train_data.head())

# Upload test data
test_data_file = st.file_uploader("Upload Test Data (CSV)", type="csv")
if test_data_file is not None:
    test_data = load_data(test_data_file)
    st.write("Test Data:")
    st.write(test_data.head())

# Button to process the data
if st.button("Process Data"):
    if train_data_file is not None and test_data_file is not None:
        st.write("Train data shape: ", train_data.shape)
        st.write("Test data shape: ", test_data.shape)
        st.success("Data loaded and ready for processing!")
    else:
        st.warning("Please upload both train and test data.")

