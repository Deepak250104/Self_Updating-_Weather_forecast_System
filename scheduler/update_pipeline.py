import os
import json
import pandas as pd
import sys
# Allow imports from project root (so utils/ can be found)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_fetcher import fetch_recent_data
from utils.feature_engineering import apply_feature_engineering
from utils.model_trainer import incremental_train_models

def update_pipeline():
    fetch_recent_data()
    
    # Load new data
    dfs = []
    for f in os.listdir("weather_data"):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join("weather_data", f))
            df['city'] = f.replace('.csv', '')
            dfs.append(df)

    new_df = pd.concat(dfs, ignore_index=True)
    new_df = apply_feature_engineering(new_df)

    acc, reg_rmses = incremental_train_models(new_df)

    # Logging
    log_path = "model/metrics_history.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append({
        "run": len(history) + 1,
        "classification_accuracy": acc,
        "regression_rmse": reg_rmses
    })

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    update_pipeline()
