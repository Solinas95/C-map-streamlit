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
    # remove last column
    RUL_data = RUL_data.drop(RUL_data.columns[-1], axis=1)
    # rename columns
    RUL_data.columns = ['RUL']
    

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




train_data = load_data('train_FD001.txt')
test_data = load_data('test_FD001.txt')
RUL_data = load_data('RUL_FD001.txt')

#RUL_data = RUL_data.drop(RUL_data.columns[1], axis=1)
#RUL_data = RUL_data.rename(columns={RUL_data.columns[0]: "RUL"})
train_data, test_data, RUL_data = preprocess_data(train_data, test_data, RUL_data)
