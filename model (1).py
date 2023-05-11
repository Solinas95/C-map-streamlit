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

    # Upload the dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write(input_df)

        # Run normalization
        normalized_df, _ = normalize_data(input_df)
        st.write("Normalized data:")
        st.write(normalized_df)

if __name__ == "__main__":
    run()
