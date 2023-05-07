#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np

st.title("CMAPSS Data Preprocessing")
st.write("Please upload your train, test and RUL data in txt format.")

def load_data(data):
    return pd.read_csv(data, delimiter=" ", header=None)


    
def preprocess_data(train_data, test_data, RUL_data):
    # Drop the last two columns from train and test data
    train_data = train_data.drop([26, 27], axis=1)
    test_data = test_data.drop([26, 27], axis=1)

    # Rename the columns
    column_names = ["unit_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    train_data.columns = column_names
    test_data.columns = column_names

    # Remove the header from the RUL data and rename the column
    # Remove the header from the RUL data and rename the column
    RUL_data = RUL_data.drop(0, axis=0)
    RUL_data['RUL'] = RUL_data[0]
    #RUL_data = RUL_data.drop(0, axis=1)
    RUL_data = RUL_data.dropna(axis=1)
    # Calculate RUL for train_data
    rul_train = pd.DataFrame(train_data.groupby("unit_id")["cycle"].max()).reset_index()
    rul_train.columns = ["unit_id", "max_cycles"]
    train_data = train_data.merge(rul_train, on=["unit_id"], how="left")
    train_data["RUL"] = train_data["max_cycles"] - train_data["cycle"]
    train_data = train_data.drop("max_cycles", axis=1)

    return train_data, test_data, RUL_data






train_data_file = st.file_uploader("Upload Train Data (txt)", type="txt")
if train_data_file is not None:
    train_data = load_data(train_data_file)
    st.write("Train Data:")
    st.write(train_data.shape)

test_data_file = st.file_uploader("Upload Test Data (txt)", type="txt")
if test_data_file is not None:
    test_data = load_data(test_data_file)
    st.write("Test Data:")
    st.write(test_data.shape)

RUL_data_file = st.file_uploader("Upload RUL Data (txt)", type="txt")
if RUL_data_file is not None:
    RUL_data = load_data(RUL_data_file)
    st.write("RUL Data:")
    st.write(RUL_data.shape)

if train_data_file is not None and test_data_file is not None and RUL_data_file is not None:
    train_data, test_data, RUL_data = preprocess_data(train_data, test_data, RUL_data)
    st.write("Preprocessed Train Data:")
    st.write(train_data.head())
    st.write("Preprocessed Test Data:")
    st.write(test_data.head())
    st.write("Preprocessed RUL Data:")
    st.write(RUL_data.head())
    


def create_X_y(data, seq_length):
    X = []
    y = []

    for unit_id in data['unit_id'].unique():
        unit_data = data[data['unit_id'] == unit_id]
        for i in range(len(unit_data) - seq_length):
            X.append(unit_data.iloc[i : i + seq_length].drop('RUL', axis=1).values)
            y.append(unit_data.iloc[i + seq_length]['RUL'])

    return X,y

def calculate_std(X, unit_id, seq_length):
    unit_X = X[unit_id]
    std_df = pd.DataFrame(unit_X).std(axis=0)
    return std_df

# Add this line after your existing file uploader widgets
create_sequences_button = st.button("Create Sequences and Labels")
# Add the slider for sequence length selection after the file uploader widgets
seq_length = st.slider("Select Sequence Length", min_value=1, max_value=100, value=50, step=1)

if create_sequences_button and train_data_file is not None:
    X, y = create_X_y(train_data, seq_length)
    st.write("Sequences and labels are created.")
    
    # Add a selectbox for the user to choose the unit_id
    unique_unit_ids = train_data["unit_id"].unique()
    selected_unit_id = st.selectbox("Select Unit ID", unique_unit_ids)

    # Add a button to show the standard deviation for the selected unit_id
    show_std_button = st.button("Show Standard Deviation")
    
    if show_std_button:
        std_df = calculate_std(X, selected_unit_id, seq_length)
        st.write(f"Standard deviation for each column in sequences of Unit ID {selected_unit_id}:")
        st.write(std_df)
