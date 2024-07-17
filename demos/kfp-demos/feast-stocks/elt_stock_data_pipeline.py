import os
import pickle
import requests
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from polygon import RESTClient
from typing import Tuple

ndx_ticker = "I:NDX"
start_date = "2024-01-01"

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
client = RESTClient(POLYGON_API_KEY)

stock_list = [
    "AAPL",
    "AMZN",
    "GOOG",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
]

data_directory = "data"


def get_daily_data(
    ticker: str = "I:NDX", start_date: str = "2024-01-01"
) -> pd.DataFrame:
    daily_ticker_data = []
    for ticker_agg in client.list_aggs(
        ticker=ticker, from_=start_date, to=todays_date, multiplier=1, timespan="day"
    ):
        daily_ticker_data.append(ticker_agg)

    df = pd.DataFrame(daily_ticker_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["timestamp"].dt.date
    return df


def get_or_load_historical_data(output_filename: str) -> Tuple[pd.DataFrame, dict]:
    if output_filename in os.listdir():
        with open("ticker_data.pkl", "rb") as output_file:
            df_dict = pickle.load(output_file)
    else:
        ndx_df = get_daily_data(ndx_ticker)
        df_dict = {ndx_ticker: ndx_df}
        for ticker in stock_list:
            df_dict[ticker] = get_daily_data(ticker)
            time.sleep(30)

        with open("ticker_data.pkl", "wb") as output_file:
            pickle.dump(df_dict, output_file)
    return ndx_df, df_dict


def generate_lag_features(df: pd.DataFrame, column: str, n_lags: int = 5) -> None:
    for t in range(1, n_lags + 1):
        df[f"{column}_lag{t}"] = df[f"{column}"].shift(t)


def calculate_sliding_window_averages(
    df: pd.DataFrame, field_name: str, max_window_size: int = 5
) -> None:
    for window_size in range(1, max_window_size + 1):
        for i in range(1, max_window_size - window_size + 1):
            column_name = f"{field_name}_window_avg_from_{i}_to_{i+window_size}"
            columns_to_window = [
                f"{field_name}_lag{c}" for c in range(i, i + window_size + 1)
            ]
            df[column_name] = df[columns_to_window].mean(axis=1)


def feature_pipeline(
    df: pd.DataFrame, features: list[str], n_lags: int = 5, max_window_size: int = 5
) -> None:
    for c in features:
        generate_lag_features(df, c, n_lags)
        calculate_sliding_window_averages(df, c, max_window_size)


def create_model_dataset(
    df: pd.DataFrame,
    main_ticker: str,
    columns_to_process: list[str],
    ticker_df_dict: dict,
    n_lags: int,
    max_window_size: int,
) -> pd.DataFrame:
    finaldf = df.copy()
    feature_pipeline(df, columns_to_process, n_lags, max_window_size)
    finaldf.columns = [f"{c}_{main_ticker.lower()}" for c in finaldf.columns]

    for ticker in ticker_df_dict:
        if ticker != main_ticker:
            feature_pipeline(
                ticker_df_dict[ticker], columns_to_process, n_lags, max_window_size
            )
            finaldf = finaldf.merge(
                ticker_df_dict[ticker].rename(
                    columns={
                        c: f"{c}_{ticker.lower()}"
                        for c in ticker_df_dict[ticker].columns
                    },
                ),
                how="left",
                left_on=f"date_{main_ticker.lower()}",
                right_on=f"date_{ticker.lower()}",
            )
    return finaldf


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_model(
    model: SimpleNN,
    features: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float = 0.1,
    num_epochs: int = 200,
):
    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete.")


def batch_score_data(model: SimpleNN, features: torch.Tensor) -> torch.Tensor:
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        predictions = model(features)
    return predictions


def main():
    todays_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
    output_filename = "ticker_data.pkl"
    columns_to_process = ["open", "low", "high"]
    n_lags, max_window_size = 5, 5
    ndx_df, df_dict = get_or_load_historical_data(
        output_filename,
        start_date,
    )
    finaldf = create_model_dataset(
        ndx_df,
        ndx_ticker,
        columns_to_process,
        df_dict,
        n_lags,
        max_window_size,
    )

    print(f"final dataset has {len(list(finaldf.columns))} columns")
    feature_list = sorted(
        [
            c
            for c in finaldf.columns
            if ("lag_" in c or "window_" in c) and "ndx" not in c
        ]
    )
    features = torch.from_numpy(
        finaldf[feature_list].values[n_lags:, :].astype(np.float32)
    )
    labels = torch.from_numpy(finaldf["open_ndx"].values[n_lags:].astype(np.float32))

    input_size = features.shape[1]  # Number of features
    hidden_size = 32  # Number of hidden units
    output_size = 1  # Number of output units
    model = SimpleNN(input_size, hidden_size, output_size)
    train_model(model, features, labels, 0.1, 200)
    predictions = batch_score_data(model, features)
    print(f"RMSE = {torch.sqrt( torch.mean( (predictions- labels) ** 2) )}")
    finaldf["predictions"] = None
    finaldf.loc[n_lags:, "predictions"] = predictions

    torch.save(model.state_dict(), "ndx_model_weights.pth")
