import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

def incremental_train_models(df, model_dir="model/"):
    # Encode classification target
    le = LabelEncoder()
    df['weather_condition_encoded'] = le.fit_transform(df['weather_condition'])

    # Common features
    common_features = ['hour', 'dayofweek', 'month', 'dayofyear', 'latitude', 'longitude']
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    df = pd.concat([df, city_dummies], axis=1)
    features = common_features + list(city_dummies.columns) + ['tempC_lag1', 'pressure_lag1', 'tempC_hourly_change', 'pressure_trend']

    ### === Classification === ###
    X_clf = df[features]
    y_clf = df['weather_condition_encoded']

    clf_model_path = os.path.join(model_dir, 'best_classification_model.pkl')
    if os.path.exists(clf_model_path):
        base_clf = joblib.load(clf_model_path)
        clf = lgb.LGBMClassifier()
        clf.fit(X_clf, y_clf, init_model=base_clf.booster_)
    else:
        clf = lgb.LGBMClassifier()
        clf.fit(X_clf, y_clf)

    clf_preds = clf.predict(X_clf)
    clf_acc = accuracy_score(y_clf, clf_preds)
    joblib.dump(clf, clf_model_path)

    ### === Regression (multi-target using independent models) === ###
    regression_targets = ['tempC', 'FeelsLikeC', 'humidity', 'windspeedKmph', 'precipMM', 'maxtempC', 'mintempC']
    reg_rmse_scores = {}
    for target in regression_targets:
        X = df[features]
        y = df[target]
        reg_model_path = os.path.join(model_dir, f'regressor_{target}.pkl')

        if os.path.exists(reg_model_path):
            base_reg = joblib.load(reg_model_path)
            reg = lgb.LGBMRegressor()
            reg.fit(X, y, init_model=base_reg.booster_)
        else:
            reg = lgb.LGBMRegressor()
            reg.fit(X, y)

        preds = reg.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        reg_rmse_scores[target] = rmse
        joblib.dump(reg, reg_model_path)

    return clf_acc, reg_rmse_scores
