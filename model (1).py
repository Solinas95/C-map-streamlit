#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, scaler=None, columns_to_exclude=['unit_ID','time_in_cycles','RUL']):
    df['cycle_norm'] = df['time_in_cycles']
    cols_normalize = df.columns.difference(columns_to_exclude)
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[cols_normalize])
    norm_df = pd.DataFrame(scaler.transform(df[cols_normalize]), 
                           columns=cols_normalize, 
                           index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns = df.columns)
    df = df.reset_index(drop=True)
    return df, scaler

def run():
    st.title("Data normalization application")

    # Load model
    json_file_path = "lstm_model.json"
    weights_file_path = "lstm_model_weights.h5"
    model = None
    if json_file_path and weights_file_path:
        model = load_keras_model(json_file_path, weights_file_path)
        st.write("Model loaded successfully on " + datetime.datetime.now().strftime("%Y-%m-%d"))

    # Advanced settings in the sidebar
    if st.sidebar.checkbox("Advanced Settings"):
        if model is not None:
            st.sidebar.write(model.summary())
        else:
            st.sidebar.write("No model loaded")

    # Upload the dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="txt")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write(input_df)

        # Check rows with null values
        null_rows = input_df.isnull().any(axis=1)
        st.write("Rows with null values:")
        st.write(input_df[null_rows])

        # Drop rows with null values
        input_df = input_df.dropna()
        st.write("Dataframe after dropping rows with null values:")
        st.write(input_df)

        # Drop columns 26 and 27
        input_df.drop(columns=[26,27], axis=1, inplace=True)
        st.write("Dataframe after dropping columns 26 and 27:")
        st.write(input_df)

        # Run normalization
        normalized_df, _ = normalize_data(input_df)
        st.write("Normalized data:")
        st.write(normalized_df)

        # You can now use the model for prediction or further analysis

if __name__ == "__main__":
    run()

