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
from typing import Tuple, Union

NDX_TICKER = "I:NDX"
start_date = "2024-01-01"
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
TODAYS_DATE = datetime.datetime.now().date().strftime("%Y-%m-%d")
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

ARCHIVE_DIR = "archive"
DATA_DIRECTORY = "data"
PREDICTIONS_DIRECTORY = "predictions"
log_file = "successful_dates.log"
model_filename = "ndx_model_weights.pth"


def get_successful_dates(log_file: str) -> set[str]:
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r") as f:
        dates = f.read().splitlines()
    return set(dates)


def log_successful_dates(dates: list[str], log_file: str) -> None:
    with open(log_file, "a") as f:
        for date in dates:
            f.write(f"{date}\n")


def get_dates_to_pull(
    start_date: str, end_date: str, successful_dates: list[str]
) -> list[str]:
    business_days = pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
    return [date for date in business_days if date not in successful_dates]


def save_data_to_parquet(df: pd.DataFrame, base_path: str):
    unique_dates = df["date_i:ndx"].astype(str).unique().tolist()
    missing_dates = [
        j for j in unique_dates if f"date_i:ndx={j}" not in os.listdir(base_path)
    ]
    if len(missing_dates) > 0:
        print(f"exporting {len(missing_dates)} more file(s)")
        dfss = df[df["date_i:ndx"].astype(str).str.contains("|".join(missing_dates))]
        dfss.to_parquet(base_path, index=False, partition_cols=["date_i:ndx"])
    else:
        print("exporting 0 files - no new data found")


def pull_stock_data(
    ticker: str = "I:NDX", start_date: str = "2024-01-01"
) -> pd.DataFrame:
    daily_ticker_data = []
    for ticker_agg in client.list_aggs(
        ticker=ticker, from_=start_date, to=TODAYS_DATE, multiplier=1, timespan="day"
    ):
        daily_ticker_data.append(ticker_agg)

    df = pd.DataFrame(daily_ticker_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["timestamp"].dt.date
    return df


def pull_all_stock_data(
    ticker_list: list[str],
    start_date: str = "2024-01-01",
    sleep_time: int = 30,
) -> pd.DataFrame:
    stock_data_dict = {}
    for ticker in ticker_list:
        stock_data_dict[ticker] = pull_stock_data(ticker, start_date)
        time.sleep(sleep_time)
    return stock_data_dict


def get_or_load_historical_data(
    output_filename: str, start_date: str = "2024-01-01"
) -> Tuple[pd.DataFrame, dict]:
    if output_filename in os.listdir():
        print("loading stored data...")
        with open(output_filename, "rb") as output_file:
            df_dict = pickle.load(output_file)
            ndx_df = df_dict[NDX_TICKER]
    else:
        print("no stored data found, calling polygon api...")
        ndx_df = pull_stock_data(NDX_TICKER, start_date)
        df_dict = pull_all_stock_data(stock_list, start_date)
        df_dict[NDX_TICKER] = ndx_df

        with open(output_filename, "wb") as output_file:
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


def run_feature_pipeline(
    df: pd.DataFrame, features: list[str], n_lags: int = 5, max_window_size: int = 5
) -> None:
    for c in features:
        generate_lag_features(df, c, n_lags)
        calculate_sliding_window_averages(df, c, max_window_size)


def create_model_dataset(
    df: pd.DataFrame,
    columns_to_process: list[str],
    ticker_df_dict: dict,
    n_lags: int,
    max_window_size: int,
) -> pd.DataFrame:
    finaldf = df.copy()
    run_feature_pipeline(df, columns_to_process, n_lags, max_window_size)
    finaldf.columns = [f"{c}_{NDX_TICKER.lower()}" for c in finaldf.columns]

    for ticker in ticker_df_dict:
        if ticker != NDX_TICKER:
            ticker_df = ticker_df_dict[ticker].copy()

            run_feature_pipeline(ticker_df, columns_to_process, n_lags, max_window_size)
            finaldf = finaldf.merge(
                ticker_df.rename(
                    columns={c: f"{c}_{ticker.lower()}" for c in ticker_df.columns},
                ),
                how="left",
                left_on=f"date_{NDX_TICKER.lower()}",
                right_on=f"date_{ticker.lower()}",
            )

    finaldf.drop_duplicates(
        subset=[f"date_{NDX_TICKER.lower()}"],
        inplace=True,
    )

    mindate = finaldf["date_i:ndx"].min()
    maxdate = finaldf["date_i:ndx"].max()
    print(f"training data ranging from {mindate} to {maxdate}")
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
        loss = criterion(outputs, labels.view(-1, 1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_filename)
    print("Training complete.")


def train_or_load_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float = 0.1,
    epochs: int = 200,
) -> SimpleNN:
    input_size = features.shape[1]  # Number of features
    hidden_size = 32  # Number of hidden units
    output_size = labels.dim()  # Number of output units
    model = SimpleNN(input_size, hidden_size, output_size)
    if model_filename in os.listdir():
        print("loading existing model...")
        weights = torch.load(model_filename)
        model.load_state_dict(weights)
    else:
        train_model(model, features, labels, learning_rate, epochs)
        torch.save(model.state_dict(), model_filename)
    return model


def batch_score_data(model: SimpleNN, features: torch.Tensor) -> torch.Tensor:
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        predictions = model(features)
    return predictions


def update_and_dedupe_full_dict(df_dict: dict, latest_df_dict: dict) -> None:
    # storing an archive in case something breaks
    with open(f"archive/ticker_data_{TODAYS_DATE}.pkl", "wb") as output_file:
        pickle.dump(df_dict, output_file)
    for ticker in df_dict:
        df_dict[ticker] = pd.concat(
            [
                df_dict[ticker],
                latest_df_dict[ticker],
            ],
            axis=0,
        )

    # over-writing the main file
    with open("ticker_data.pkl", "wb") as output_file:
        pickle.dump(df_dict, output_file)


def get_latest_data(
    max_date_val: datetime.date,
    df_dict: dict,
    columns_to_process: list[str],
    n_lags: int,
    max_window_size: int,
) -> Union[None, pd.DataFrame]:
    yesterdays_date_val = datetime.date.today() - datetime.timedelta(days=1)
    if max_date_val != yesterdays_date_val:
        maxdate = f"{max_date_val}"
        print(f"getting latest data starting from {maxdate}...")
        latest_df_dict = pull_all_stock_data(
            [NDX_TICKER] + stock_list, start_date=maxdate
        )
        update_and_dedupe_full_dict(df_dict, latest_df_dict)
        ndx_df_new = latest_df_dict[NDX_TICKER]
        del latest_df_dict[NDX_TICKER]
        newdf = create_model_dataset(
            ndx_df_new,
            columns_to_process,
            latest_df_dict,
            n_lags,
            max_window_size,
        )
        save_data_to_parquet(newdf, f"{DATA_DIRECTORY}/")

        loaded_successful_dates = newdf["date_i:ndx"].astype(str).tolist()
        log_successful_dates(loaded_successful_dates, log_file)
        return newdf


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    os.makedirs(PREDICTIONS_DIRECTORY, exist_ok=True)
    successful_dates = get_successful_dates(log_file)
    dates_to_pull = get_dates_to_pull(start_date, TODAYS_DATE, successful_dates)

    output_filename = "ticker_data.pkl"
    columns_to_process = ["open", "low", "high"]
    n_lags, max_window_size = 5, 5

    ndx_df, df_dict = get_or_load_historical_data(
        output_filename,
        start_date,
    )

    finaldf = create_model_dataset(
        ndx_df,
        columns_to_process,
        df_dict,
        n_lags,
        max_window_size,
    )
    save_data_to_parquet(finaldf, f"{DATA_DIRECTORY}/")

    maxdate_val = finaldf["date_i:ndx"].max()
    newdf = get_latest_data(
        maxdate_val, df_dict, columns_to_process, n_lags, max_window_size
    )
    if newdf is not None:
        finaldf = pd.concat([finaldf, newdf], axis=0)

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
    labels = torch.from_numpy(finaldf["open_i:ndx"].values[n_lags:].astype(np.float32))
    model = train_or_load_model(features, labels, 0.1, 200)

    predictions = batch_score_data(model, features)
    print(f"RMSE = {torch.sqrt( torch.mean( (predictions- labels) ** 2) )}")

    finaldf["predictions"] = None
    finaldf.loc[n_lags:, "predictions"] = predictions
    finaldf["run_date"] = TODAYS_DATE

    prediction_df_columns_to_save = [
        "date_i:ndx",
        "open_i:ndx",
        "predictions",
        "run_date",
    ]
    save_data_to_parquet(
        finaldf[prediction_df_columns_to_save], f"{PREDICTIONS_DIRECTORY}/"
    )
    print(f"latest predictions:\n{finaldf[prediction_df_columns_to_save].tail()}")


if __name__ == "__main__":
    main()
