#!/usr/bin/env python
# coding: utf-8

# In[33]:


import io
import pandas as pd
import requests
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Define the available configurations
configurations = ["Config 1", "Config 2", "Config 3", "Config 4"]

# Define a function to load a dataframe from a URL
def load_df_from_url(url):
    response = requests.get(url)
    file_content = response.content.decode("ascii")
    df = pd.read_csv(io.StringIO(file_content), sep=" ", header=None)
    return df

# Streamlit app
#st.title("Data Loading Tool")

# Allow the user to select a configuration
selected_config = st.selectbox("Select a configuration", options=configurations)

# Load the corresponding dataset from Kaggle
if selected_config == "Config 1":
    url_train = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/train_FD001.txt"
    url_test = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/test_FD001.txt"
    url_rul = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/RUL_FD001.txt"
elif selected_config == "Config 2":
    url_train = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/train_FD002.txt"
    url_test = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/test_FD002.txt"
    url_rul = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/RUL_FD002.txt"
elif selected_config == "Config 3":
    url_train = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/train_FD003.txt"
    url_test = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/test_FD003.txt"
    url_rul = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/RUL_FD003.txt"
elif selected_config == "Config 4":
    url_train = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/train_FD004.txt"
    url_test = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/test_FD004.txt"
    url_rul = "https://raw.githubusercontent.com/ovbystrova/C-MAPSS-problem/main/RUL_FD004.txt"

df_train = load_df_from_url(url_train)
df_test = load_df_from_url(url_test)
df_RUL = load_df_from_url(url_rul)

# Remove columns with NaN values
df_train.drop(columns=[26,27], axis=1, inplace=True)
df_test.drop(columns=[26,27], axis=1, inplace=True)
df_RUL.drop(columns=[1], axis=1, inplace=True)

# Rename columns
columns_train = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]

df_train.columns = columns_train
df_test.columns = columns_train

# Define a function to show the plot of the number of cycles per unit
def show_plot():
    # Load the dataframes for the selected configuration
    if selected_config == "Config 1":
        url_train = "https://raw.githubusercontent.com/baymax-uikit/C-maps-data/main/train_FD001.txt"
    elif selected_config == "Config 2":
        url_train = "https://raw.githubusercontent.com/baymax-uikit/C-maps-data/main/train_FD002.txt"
    elif selected_config == "Config 3":
        url_train = "https://raw.githubusercontent.com/baymax-uikit/C-maps-data/main/train_FD003.txt"
    elif selected_config == "Config 4":
        url_train = "https://raw.githubusercontent.com/baymax-uikit/C-maps-data/main/train_FD004.txt"
    df_train = load_df_from_url(url_train)

    # Clean the data
    df_train.drop(columns=[26, 27], axis=1, inplace=True)
    columns_train = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf',
               'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
    df_train.columns = columns_train

    # Show the plot of the number of cycles per unit
    cnt_train = df_train[["unit_ID", "time_in_cycles"]].groupby("unit_ID").max().sort_values(by="time_in_cycles", ascending=False)
    cnt_ind = [str(i) for i in cnt_train.index.to_list()]
    cnt_val = list(cnt_train.time_in_cycles.values)

    plt.style.use("seaborn")
    plt.figure(figsize=(12, 30))
    sns.barplot(x=list(cnt_val), y=list(cnt_ind), palette='Spectral')
    plt.xlabel('Number of Cycles')
    plt.ylabel('Unit ID')
    plt.title('Number of Cycles per Unit', fontweight='bold', fontsize=24, pad=15)
    
    # Return the plot
    return plt


# Streamlit app
st.title("Data Loading Tool")

# Allow the user to select a configuration
selected_config = st.selectbox("Select a configuration", options=configurations)

# Show the plot of the number of cycles per unit when clicked
if st.button("Show Plot"):
    fig = show_plot()
    st.pyplot(fig)    

