# 🌤️ Self-Updating Weather Forecast System

A fully automated, incrementally learning weather prediction system for Indian cities that:
- Fetches new weather data every 15 days via **WorldWeatherOnline API**
- Uses **incremental training** to improve both classification and regression LightGBM models over time
- Automatically retrains, evaluates, and updates models using **GitHub Actions**
- Tracks performance (accuracy & RMSE) over time via a persistent log
- Requires no manual intervention once deployed

---

## 📦 Dataset

Historical dataset used to bootstrap the models:

**[📥 Kaggle: Historical Weather Data for Indian Cities](https://www.kaggle.com/datasets/hiteshsoneji/historical-weather-data-for-indian-cities)**

> Download this zip manually and extract the CSVs into the `weather_data/` folder like so:

```
weather_data/
├── bengaluru.csv
├── bombay.csv
├── delhi.csv
├── hyderabad.csv
├── jaipur.csv
├── kanpur.csv
├── nagpur.csv
├── pune.csv
```

---

## 🧠 Features & Model Architecture

- **Classification Target**: Predicts weather type (`sunny`, `rainy`, `cloudy`, etc.)
- **Regression Targets**: Forecasts multiple values such as `tempC`, `humidity`, `windspeed`, etc.
- **Feature Engineering**: Includes lagged features, temporal patterns, location coordinates, and weather condition rules
- **Incremental Learning**: Uses LightGBM’s `init_model` for continuous training
- **Performance Logging**: Stores accuracy and RMSE in `model/metrics_history.json` after every training cycle

---

## ⚙️ Project Structure

```
SELF_UPDATING--WEATHER_FORECAST-SYSTEM/
├── model/
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   └── metrics_history.json         
│
├── notebook/
│   └── Weather_forecast.ipynb
│
├── scheduler/
│   └── update_pipeline.py           
│
├── utils/
│   ├── data_fetcher.py               
│   └── feature_engineering.py        
│   └── model_trainer.py              
│
├── weather_data/
│   ├── bengaluru.csv
│   ├── ...
│
├── .github/
│   └── workflows/
│       └── update_weather.yml        
│
├── requirements.txt
├── README.md

```

---

## 🛰️ Automation & Scheduling

### 🔁 Workflow: `update_pipeline.py`

This script:
1. Fetches 15 days of hourly data for 8 Indian cities from the **WorldWeatherOnline API** using `wwo_hist`
2. Applies the same feature engineering as used in the original notebook
3. Loads existing models and **incrementally trains** them using LightGBM
4. Logs accuracy & RMSE to `model/metrics_history.json`
5. Saves updated models to disk

> ✅ Run it manually:
```bash
python scheduler/update_pipeline.py
```

---

## ⏱️ GitHub Actions Automation

### ✅ File: `.github/workflows/update_weather.yml`

- The pipeline runs **every 15 days** automatically.
- It does everything the local pipeline does — fetch, retrain, evaluate, and update the models in GitHub.

> ✅ To trigger manually:
- Go to **Actions → Weather Model Auto Update → Run workflow**

---

## 🧪 Setup Locally

> You should use Python 3.10 and Pandas 1.5.3 (due to compatibility with `wwo_hist`)

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # (Windows PowerShell)

# Install dependencies
pip install -r requirements.txt
```

---

## 📈 Performance Logs

Model metrics are automatically saved to:
```bash
model/metrics_history.json
```

Example:
```json
[
  {
    "run": 1,
    "classification_accuracy": 0.61,
    "regression_rmse": {
      "tempC": 2.1,
      "humidity": 9.8,
      ...
    }
  },
  ...
]