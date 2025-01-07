# Weather Forecasting App

## Overview
The **Weather Forecasting App** is a Streamlit-based application that predicts weather conditions (daily high/low temperatures and hourly temperatures) for a given location using pre-trained LSTM models. Users can input a location to get forecasts based on historical weather data fetched from the Meteostat API.

---

## Features
1. **Location-based Forecasting**:
   - Users enter a location to fetch weather predictions.
   - Uses the Geopy library to convert the location name to geographical coordinates (latitude and longitude).

2. **Daily High/Low Forecasting**:
   - Predicts daily maximum and minimum temperatures for the next 7 days.
   - Displays results in a table and a Plotly chart for visualization.

3. **Hourly Temperature Forecasting**:
   - Predicts hourly temperature for the next 7 hours.
   - Displays results in a table and a Plotly chart for visualization.

4. **Data Processing**:
   - Historical weather data is fetched and preprocessed from the Meteostat API.
   - Data normalization is performed using MinMaxScaler for accurate LSTM model predictions.

5. **LSTM Models**:
   - Pre-trained models (`daily_data_model.keras` and `hourly_data_model.keras`) are used for recursive forecasting.

---

## Installation

### Prerequisites
- Python 3.8 or later
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd weather-forecasting-app
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained LSTM models and place them in the project directory:
   - `daily_data_model.keras`
   - `hourly_data_model.keras`

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Libraries Used
- **Streamlit**: For building the web app UI.
- **Geopy**: To fetch geographical coordinates from user input.
- **Meteostat**: To retrieve historical weather data.
- **TensorFlow**: To load and predict using LSTM models.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For data scaling.
- **Plotly**: For interactive and dynamic plotting.

---

## File Structure
```
weather-forecasting-app/
├── app.py               # Main application script
├── daily_data_model.keras  # Pre-trained LSTM model for daily forecasting
├── hourly_data_model.keras # Pre-trained LSTM model for hourly forecasting
├── requirements.txt      # List of dependencies
├── README.md            # Project documentation
```

---

## Usage
1. Open the app in a web browser after starting Streamlit.
2. Enter a location (e.g., "Vancouver, Canada") in the input box.
3. Click the **Submit** button to fetch the weather forecast.
4. View the following results:
   - **Daily Forecast**: A table and line chart displaying high and low temperatures for the next 7 days.
   - **Hourly Forecast**: A table and line chart showing temperatures for the next 7 hours.

---

## Example
### Input
- Location: "Delhi, India"

### Output
- **Daily Forecast**:
  - High and low temperatures for the next 7 days.
  - Line chart of daily high and low temperatures.

- **Hourly Forecast**:
  - Hourly temperature predictions for the next 7 hours.
  - Line chart of hourly temperature trends.

---

## Future Improvements
1. Extend forecasting to include additional weather parameters (e.g., precipitation, wind speed).
2. Add user-configurable forecasting intervals (e.g., 3-day or 24-hour forecasts).
3. Improve UI/UX with additional charts and interactivity.

---



