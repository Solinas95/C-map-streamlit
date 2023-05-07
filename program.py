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
    RUL_data = RUL_data.drop(0, axis=0)
    RUL_data['RUL'] = RUL_data[0]
    RUL_data = RUL_data.drop(0, axis=1)



    return train_data, test_data, RUL_data



    
def preprocess_data(train_data, test_data, RUL_data):
    # Drop the last two columns from train and test data
    train_data = train_data.drop([26, 27], axis=1)
    test_data = test_data.drop([26, 27], axis=1)

    # Rename the columns
    column_names = ["unit_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    train_data.columns = column_names
    test_data.columns = column_names

    # Remove the header from the RUL data and rename the column
    RUL_data = RUL_data.drop(0, axis=0)
    RUL_data.columns = ["RUL"]

    # Calculate RUL for train_data
    rul_train = pd.DataFrame(train_data.groupby("unit_id")["cycle"].max()).reset_index()
    rul_train.columns = ["unit_id", "max_cycles"]
    train_data = train_data.merge(rul_train, on=["unit_id"], how="left")
    train_data["RUL"] = train_data["max_cycles"] - train_data["cycle"]
    train_data = train_data.drop("max_cycles", axis=1)

    return train_data, test_data, RUL_data


def create_X_y(train_data, seq_length):
    train_data = train_data.drop("unit_id", axis=1)
    
    X_train, y_train = [], []

    unit_ids = train_data["unit_id"].unique()
    
    for unit_id in unit_ids:
        unit_data = train_data[train_data["unit_id"] == unit_id]
        unit_data = unit_data.drop("unit_id", axis=1)
        for i in range(len(unit_data) - seq_length):
            X_train.append(unit_data.iloc[i:i + seq_length, 1:].values)
            y_train.append(unit_data.iloc[i + seq_length, -1])

    return np.array(X_train), np.array(y_train)



train_data_file = st.file_uploader("Upload Train Data (txt)", type="txt")
if train_data_file is not None:
    train_data = load_data(train_data_file)
    st.write("Train Data:")
    st.write(train_data.head())

test_data_file = st.file_uploader("Upload Test Data (txt)", type="txt")
if test_data_file is not None:
    test_data = load_data(test_data_file)
    st.write("Test Data:")
    st.write(test_data.head())

RUL_data_file = st.file_uploader("Upload RUL Data (txt)", type="txt")
if RUL_data_file is not None:
    RUL_data = load_data(RUL_data_file)
    st.write("RUL Data:")
    st.write(RUL_data.head())

if train_data_file is not None and test_data_file is not None and RUL_data_file is not None:
    train_data, test_data, RUL_data = preprocess_data(train_data, test_data, RUL_data)
    st.write("Preprocessed Train Data:")
    st.write(train_data.head())
    st.write("Preprocessed Test Data:")
    st.write(test_data.head())
    st.write("Preprocessed RUL Data:")
    st.write(RUL_data.head())
    
# Set the sequence length for the LSTM
seq_length = 50

# Split the train data into X and y
#X_train, y_train = create_X_y(train_data, seq_length)

