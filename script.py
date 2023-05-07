#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title("RUL Prediction using LSTM")
st.write("Please upload your train and test data in CSV format.")

def load_data(data):
    return pd.read_csv(data)

train_data_file = st.file_uploader("Upload Train Data (CSV)", type="csv")
if train_data_file is not None:
    train_data = load_data(train_data_file)
    st.write("Train Data:")
    st.write(train_data.head())

test_data_file = st.file_uploader("Upload Test Data (CSV)", type="csv")
if test_data_file is not None:
    test_data = load_data(test_data_file)
    st.write("Test Data:")
    st.write(test_data.head())

def preprocess_data(train_data, seq_length):
    train_data = train_data.drop("unit_id", axis=1)
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    X_train, y_train = [], []

    for i in range(seq_length, len(train_data_scaled)):
        X_train.append(train_data_scaled[i-seq_length:i, :-1])
        y_train.append(train_data_scaled[i, -1])

    return np.array(X_train), np.array(y_train), scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_lstm_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

def evaluate_lstm_model(model, X_train, y_train):
    train_loss = model.evaluate(X_train, y_train, verbose=1)
    return train_loss

def predict_lstm(model, test_data, scaler, seq_length):
    test_data = test_data.drop("unit_id", axis=1)
    test_data_scaled = scaler.transform(test_data)

    X_test = []
    for i in range(seq_length, len(test_data_scaled)):
        X_test.append(test_data_scaled[i-seq_length:i, :])

    X_test = np.array(X_test)
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], y_pred), axis=1))[:, -1]

    return y_pred_inverse

def plot_predicted_rul(unit_id, y_pred, test_data):
    unit_ids = test_data["unit_id"].unique()
    if unit_id in unit_ids:
        unit_mask = test_data["unit_id"] == unit_id
        unit_test_data = test_data[unit_mask]
        unit_y_pred = y_pred[:unit_test_data.shape[0]-seq_length]

        plt.figure(figsize=(10, 5))
        plt.plot(unit_y_pred, label="Predicted RUL")
        plt.xlabel("Sequence")
        plt.ylabel("RUL")
        plt.title(f"Predicted RUL for Unit ID {unit_id}")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Invalid unit_id")

seq_length = 50

if train_data_file is not None and test_data_file is not None:
   


# In[1]:



import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title("RUL Prediction using LSTM")
st.write("Please upload your train and test data in CSV format.")

def load_data(data):
    return pd.read_csv(data)

train_data_file = st.file_uploader("Upload Train Data (CSV)", type="csv")
if train_data_file is not None:
    train_data = load_data(train_data_file)
    st.write("Train Data:")
    st.write(train_data.head())

test_data_file = st.file_uploader("Upload Test Data (CSV)", type="csv")
if test_data_file is not None:
    test_data = load_data(test_data_file)
    st.write("Test Data:")
    st.write(test_data.head())

def preprocess_data(train_data, seq_length):
    train_data = train_data.drop("unit_id", axis=1)
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    X_train, y_train = [], []

    for i in range(seq_length, len(train_data_scaled)):
        X_train.append(train_data_scaled[i-seq_length:i, :-1])
        y_train.append(train_data_scaled[i, -1])

    return np.array(X_train), np.array(y_train), scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_lstm_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

def evaluate_lstm_model(model, X_train, y_train):
    train_loss = model.evaluate(X_train, y_train, verbose=1)
    return train_loss

def predict_lstm(model, test_data, scaler, seq_length):
    test_data = test_data.drop("unit_id", axis=1)
    test_data_scaled = scaler.transform(test_data)

    X_test = []
    for i in range(seq_length, len(test_data_scaled)):
        X_test.append(test_data_scaled[i-seq_length:i, :])

    X_test = np.array(X_test)
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], y_pred), axis=1))[:, -1]

    return y_pred_inverse

def plot_predicted_rul(unit_id, y_pred, test_data):
    unit_ids = test_data["unit_id"].unique()
    if unit_id in unit_ids:
        unit_mask = test_data["unit_id"] == unit_id
        unit_test_data = test_data[unit_mask]
        unit_y_pred = y_pred[:unit_test_data.shape[0]-seq_length]

        plt.figure(figsize=(10, 5))
        plt.plot(unit_y_pred, label="Predicted RUL")
        plt.xlabel("Sequence")
        plt.ylabel("RUL")
        plt.title(f"Predicted RUL for Unit ID {unit_id}")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Invalid unit_id")

seq_length = 50

if train_data_file is not None and test_data_file is not None:
   
    # Preprocess the data
    X_train, y_train, scaler = preprocess_data(train_data, seq_length)

    # Build and train the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape)
    train_lstm_model(lstm_model, X_train, y_train, epochs=100, batch_size=64)

    # Evaluate the model
    train_loss = evaluate_lstm_model(lstm_model, X_train, y_train)
    st.write(f"Train Loss (MSE): {train_loss:.4f}")

    # Make predictions on the test data
    y_pred = predict_lstm(lstm_model, test_data, scaler, seq_length)
    st.write("Predicted RUL values:")
    st.write(y_pred)

    # Get unique unit_ids
    unique_unit_ids = test_data["unit_id"].unique()
    selected_unit_id = st.selectbox("Select a Unit ID", unique_unit_ids)

    # Plot the predicted RUL for the selected unit_id
    plot_predicted_rul(selected_unit_id, y_pred, test_data)


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

