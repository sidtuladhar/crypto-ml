import math

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
)

from src.scraper import load_csv

config = {"mode": "classification"}


def classify_change(pct_change, buy_threshold=3.0, sell_threshold=-3.0):
    if pct_change > buy_threshold:
        return "Buy"
    elif pct_change < sell_threshold:
        return "Sell"
    else:
        return "Hold"


def init_data(file_path):
    df = load_csv(file_path)
    df["Volume"] = df["Volume"].str.replace(",", "").astype(int)
    df["Pct_Change"] = ((df["Close"] - df["Close"].shift(-1)) / df["Close"]) * 100
    df["Pct_Change_5d"] = (df["Close"] - df["Close"].shift(-5)) / df["Close"] * 100
    df["Signal"] = df["Pct_Change_5d"].apply(classify_change)
    df["Lag_1"] = df["Close"].shift(-1)
    df["Lag_2"] = df["Close"].shift(-2)

    df = df.iloc[::-1]
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    # Voltaility
    df["Volatility_10d"] = df["Pct_Change"].rolling(window=10).std()
    df["Lag_1_MA_5"] = df["Lag_1"] * df["MA_5"]

    # Bollinger Bands
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = df["SMA_20"] + (2 * df["Std_Dev"])
    df["Bollinger_Lower"] = df["SMA_20"] - (2 * df["Std_Dev"])
    df = df.iloc[::-1]
    df.dropna(inplace=True)

    # NOTE: data order is earliest to oldest
    test_data = df[: math.floor(len(df) * 0.2)]
    training_data = df[math.floor(len(df) * 0.2) :]

    return training_data, test_data


if __name__ == "__main__":
    training_data, test_data = init_data("./data/shopify_data.csv")
    print(test_data.head(5))
    if config["mode"] == "regression":
        regression_y_train = training_data["Close"]

        regression_x_train = training_data.drop(
            columns=["Close", "Date", "Signal", "Pct_Change", "Pct_Change_5d"]
        )
        regression_x_train = regression_x_train.dropna()

        regression_y_test = test_data["Close"]

        regression_x_test = test_data.drop(
            columns=["Close", "Date", "Signal", "Pct_Change", "Pct_Change_5d"]
        )
        regressor = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )

        regressor.fit(regression_x_train, regression_y_train)

        regression_y_pred = regressor.predict(regression_x_test)

        mse = mean_squared_error(regression_y_test, regression_y_pred)
        rmse = mse**0.5
        rmse_percent = (rmse / test_data["Close"].mean()) * 100
        print(f"RMSE %: {rmse_percent:.2f}")

    elif config["mode"] == "classification":
        classification_y_train = training_data["Signal"]

        classification_x_train = training_data.drop(
            columns=["Signal", "Date", "Close", "Pct_Change", "Pct_Change_5d"]
        ).dropna()
        classification_x_train = classification_x_train.dropna()

        classification_y_test = test_data["Signal"]

        classification_x_test = test_data.drop(
            columns=["Signal", "Date", "Close", "Pct_Change", "Pct_Change_5d"]
        ).dropna()

        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=52, class_weight="balanced"
        )

        # Train the model
        model.fit(classification_x_train, classification_y_train)

        # Predict on the test set
        classification_y_pred = model.predict(classification_x_test)

        comparison = pd.DataFrame(
            {
                "Actual": classification_y_test.reset_index(drop=True),
                "Predicted": pd.Series(classification_y_pred),
            }
        )

        print(
            "Training Score:",
            model.score(classification_x_train, classification_y_train),
        )
        print("Test Score:", model.score(classification_x_test, classification_y_test))

        print(comparison.head(100))
        # Evaluate performance
        report = classification_report(classification_y_test, classification_y_pred)
        print("Classification Report:")
        print(report)
