from wwo_hist import retrieve_hist_data
from datetime import datetime, timedelta
import os

CITIES = ["bengaluru", "bombay", "delhi", "hyderabad", "jaipur", "kanpur", "nagpur", "pune"]

def fetch_recent_data(output_dir="weather_data"):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    for city in CITIES:
        print(f"Fetching {city}")
        retrieve_hist_data(
            output_dir,
            city,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            frequency=1,
            location_label=False,
            export_csv=True,
            store_df=False
        )
