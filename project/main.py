import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def load_data():
    df = pd.read_csv("data/covid_data.csv")
    df = df.drop(columns=["Province/State", "Lat", "Long"])
    df = df.groupby("Country/Region").sum()
    return df

def get_country_data(df, country):
    data = df.loc[country]
    data = data.T

    data.index = pd.to_datetime(data.index, format="%m/%d/%y")
    data = data.reset_index()
    data.columns = ["Date", "Cases"]

    data["Daily"] = data["Cases"].diff().fillna(0)
    data["Growth"] = data["Cases"].pct_change().fillna(0)

    data["Days"] = np.arange(len(data))

    # 🔥 NEW FEATURES (important)
    data["Lag1"] = data["Cases"].shift(1).fillna(0)
    data["Lag2"] = data["Cases"].shift(2).fillna(0)

    return data

def predict_cases(data, days_ahead):
    X = data[["Days", "Lag1", "Lag2"]]
    y = data["Cases"]

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)

    future = []
    last_vals = data.tail(2)["Cases"].values.tolist()

    for i in range(days_ahead):
        day = len(data) + i
        lag1 = last_vals[-1]
        lag2 = last_vals[-2] if len(last_vals) > 1 else last_vals[-1]

        pred = model.predict([[day, lag1, lag2]])[0]
        future.append(pred)

        last_vals.append(pred)

    return future


def calculate_accuracy(data):
    X = data[["Days", "Lag1", "Lag2"]]
    y = data["Cases"]

    split = int(len(data) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    return mae