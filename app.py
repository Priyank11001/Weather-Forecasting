import streamlit as st
from geopy.geocoders import Nominatim
from meteostat import Point, Daily, Hourly
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


# Function to get coordinates from user input
def get_coordinates(location):
    geolocator = Nominatim(user_agent="weather_forecasting_app")
    location = geolocator.geocode(location)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None


def load_and_preprocess_daily_data(latitude, longitude, is_daily=True):
    start_date = pd.to_datetime("2020-01-01")
    end_date = pd.to_datetime("today")
    location_point = Point(latitude, longitude, 70)

    # Fetch data from Meteostat
    data = Daily(location_point, start_date, end_date)
    df = data.fetch()
    df.dropna(subset=["tmax", "tmin"], inplace=True)

    df.reset_index(inplace=True)
    time = pd.to_datetime(df["time"])

    df.drop(columns=["time", "tavg", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"], axis=1, inplace=True)
   
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return scaled_data, time, scaler

def load_and_preprocess_hourly_data(latitude, longitude):
    start_date = pd.to_datetime("2020-01-01")
    end_date = pd.to_datetime("today")
    location_point = Point(latitude, longitude, 70)

    # Fetch data from Meteostat
    data = Hourly(location_point, start_date, end_date) 
    df = data.fetch()
    df.reset_index(inplace=True)
    df.dropna(subset=["temp"], inplace=True)
    time = pd.to_datetime(df["time"])
    df.drop(columns = ['time','dwpt','rhum','prcp','snow','wdir','wspd','wpgt','pres','tsun','coco'], axis=1, inplace=True)
  
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    return scaled_data, time, scaler


# Function to perform recursive forecasting
def recursive_forecasting(model, input_data, forecast_steps, timestep):
    predictions = []
    current_input = input_data[-timestep:]  # Use the last `timestep` data points

    for _ in range(forecast_steps):
        # Reshape to (1, timestep, features)
        current_input_reshaped = np.expand_dims(current_input, axis=0)
        pred = model.predict(current_input_reshaped, verbose=0)

        # Append the prediction
        predictions.append(pred[0])

        # Update current_input for the next timestep
        current_input = np.vstack((current_input[1:], pred))

    return np.array(predictions)


# Streamlit UI
st.title("Weather Forecasting App 🌤️")
st.write("Predict weather conditions based on location and LSTM models.")

# User input for location
user_location = st.text_input("Enter Location", placeholder="e.g., Vancouver, Canada")

if st.button("Submit"):
    if not user_location:
        st.error("Please enter a valid location.")
    else:
        # Get coordinates
        latitude, longitude = get_coordinates(user_location)
        if latitude is None or longitude is None:
            st.error("Unable to fetch coordinates for the given location. Please try again.")
        else:
            # st.success(f"Coordinates: Latitude {latitude}, Longitude {longitude}")

            # Daily Forecasting
            st.subheader("Daily High/Low Forecasting")
            scaled_data_daily, time_daily, scaler_daily = load_and_preprocess_daily_data(latitude, longitude, is_daily=True)
            daily_model = load_model("daily_data_model.keras")
            daily_predictions = recursive_forecasting(daily_model, scaled_data_daily, 7, 100)  # Fixed 7 days
            daily_predictions = scaler_daily.inverse_transform(daily_predictions)

            # Display results using Plotly for daily forecasting
            high_low_df = pd.DataFrame(daily_predictions, columns=["High", "Low"], index=pd.date_range(start=time_daily.iloc[-1] + pd.Timedelta(days=1), periods=7))
            

            # Plot using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=high_low_df.index, y=high_low_df["High"], mode='lines+markers', name='High'))
            fig.add_trace(go.Scatter(x=high_low_df.index, y=high_low_df["Low"], mode='lines+markers', name='Low'))
            fig.update_layout(title='Daily High/Low Temperature Forecast',
                              xaxis_title='Date',
                              yaxis_title='Temperature (°C)')
            st.plotly_chart(fig)

            # Hourly Forecasting
            st.subheader("Hourly Temperature Forecasting")
            scaled_data_hourly, time_hourly, scaler_hourly = load_and_preprocess_hourly_data(latitude, longitude)
            hourly_model = load_model("hourly_data_model.keras")
            hourly_predictions = recursive_forecasting(hourly_model, scaled_data_hourly, 7, 100)  # Fixed 7 hours
            hourly_predictions = scaler_hourly.inverse_transform(hourly_predictions)

            # Display results using Plotly for hourly forecasting
            hourly_df = pd.DataFrame(hourly_predictions, columns=["Temperature"], index=pd.date_range(start=time_hourly.iloc[-1] + pd.Timedelta(hours=1), periods=7, freq="H"))
            

            # Plot using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["Temperature"], mode='lines+markers', name='Temperature'))
            fig.update_layout(title='Hourly Temperature Forecast',
                              xaxis_title='Time',
                              yaxis_title='Temperature (°C)')
            st.plotly_chart(fig)
