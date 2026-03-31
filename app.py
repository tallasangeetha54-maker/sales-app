import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("📊 Sales Forecasting App")
st.write("App is running...")

file = st.file_uploader("Upload any CSV", type=["csv"])

if file is not None:
    st.write("File uploaded successfully!")

    df = pd.read_csv(file)
    
    st.write("Preview of data:")
    st.write(df.head())
    
    #Select columns
    date_col = st.selectbox("Select Date column", df.columns)
    sales_col = st.selectbox("Select Sales column", df.columns)

   
    df = df[[date_col, sales_col]]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.groupby(date_col).sum().reset_index()
    df = df.rename(columns={date_col: 'ds', sales_col: 'y'})
    st.write("Total Sales:", df['y'].sum())
    st.write("Highest Sales:", df['y'].max())

    days = st.slider("Select number of days to forecast", 30, 180)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    st.success("Forecast generated successfully!")

    st.subheader("📈 Forecast Graph")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("📊 Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # ✅ MOVE THESE INSIDE
    st.subheader("📄 Data Preview")
    st.write(df.head())

    st.subheader("⬇ Download Forecast")
    csv = forecast.to_csv(index=False)
    st.download_button("Download CSV", csv, "forecast.csv")

    st.subheader("📢 Insights")
    st.write("Sales are expected to increase in upcoming days based on trend.")
    