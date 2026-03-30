import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("📊 Sales Forecasting App")
st.write("App is running...")

file = st.file_uploader("Upload your Walmart Sales CSV", type=["csv"])

if file is not None:
    st.write("File uploaded successfully!")

    df = pd.read_csv(file)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Weekly_Sales']]
    df = df.groupby('Date').sum().reset_index()
    df = df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

    days = st.slider("Select number of days to forecast", 30, 180)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

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