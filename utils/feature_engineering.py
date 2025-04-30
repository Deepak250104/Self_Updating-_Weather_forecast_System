import pandas as pd
import numpy as np

def apply_feature_engineering(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by='date_time')

    # Lags and trends
    df['tempC_lag1'] = df.groupby('city')['tempC'].shift(1)
    df['pressure_lag1'] = df.groupby('city')['pressure'].shift(1)
    df['tempC_hourly_change'] = df.groupby('city')['tempC'].diff()
    df['pressure_trend'] = df.groupby('city')['pressure'].diff(3)
    df.fillna(0, inplace=True)

    # Temporal
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['dayofyear'] = df['date_time'].dt.dayofyear

    # Coordinates
    city_coordinates = {
        'delhi': (28.6273928, 77.1716954),
        'nagpur': (21.1498134, 79.0820556),
        'kanpur': (26.4609135, 80.3217588),
        'bombay': (19.054999, 72.8692035),
        'bengaluru': (12.971599, 77.594566),
        'pune': (18.5213738, 73.8545071),
        'hyderabad': (17.360589, 78.4740613),
        'jaipur': (26.9154576, 75.8189817),
    }

    df['latitude'] = df['city'].map(lambda c: city_coordinates[c][0])
    df['longitude'] = df['city'].map(lambda c: city_coordinates[c][1])

    # Weather condition label
    def categorize_weather(row):
        if row['precipMM'] > 1:
            return 'rainy'
        elif row['totalSnow_cm'] > 0.1:
            return 'snowy'
        elif row['cloudcover'] < 10:
            return 'clear'
        elif row['cloudcover'] < 50:
            return 'sunny'
        else:
            return 'cloudy'

    df['weather_condition'] = df.apply(categorize_weather, axis=1)

    return df
