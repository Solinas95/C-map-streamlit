#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.metrics import MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("RUL Prediction using LSTM")
st.write("Please upload your train and test data in CSV format.")

def load_data(data):
    return pd.read_csv(data)

def preprocess_data(train_data):
    train_data = train_data.drop("unit_ID", axis=1)
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    return train_data_scaled, scaler

def train_val_split(X, y, test_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val

def create_sequences(data, seq_length, target_column_index):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, target_column_index])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=[MeanAbsoluteError()])
    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val))
    return history

train_data_file = st.file_uploader("Upload Train Data (CSV)", type="csv")
if train_data_file is not None:
    train_data = load_data(train_data_file)
    st.write("Train Data:")
    st.write(train_data.head())

seq_length = 50

if train_data_file is not None:
    # Preprocess the data
    train_data_scaled, scaler = preprocess_data(train_data)

    # Create sequences
    X, y = create_sequences(train_data_scaled, seq_length, -1)

    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # Build and train the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape)
    history = train_lstm_model(lstm_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64)

    # Check if the user wants to see the training history plot
    if st.button("Show Training History"):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['mean_absolute_error'], label='Training MAE')
        plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training History')
        plt.legend()
        st.pyplot(plt)

