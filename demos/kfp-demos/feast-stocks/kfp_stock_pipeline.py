from kfp import dsl
from kfp.local import init, SubprocessRunner
import os
import sys
import subprocess

# Initialize SubprocessRunner
init(runner=SubprocessRunner(use_venv=False), pipeline_root='./local_outputs')

# Define component functions

@dsl.component
def fetch_stock_data(api_key: str, output_dir: str):
    import sys
    import subprocess
    import time
    import datetime
    import os
    import pickle
    import pandas as pd
    from polygon import RESTClient

    print('fetching stock data...')
    ndx_ticker = "I:NDX"
    start_date = "2024-01-01"
    todays_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
    client = RESTClient(api_key)
    stock_list = [
        "AAPL", "AMZN", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA",
    ]

    def pull_stock_data(ticker: str, start_date: str) -> pd.DataFrame:
        daily_ticker_data = []
        for ticker_agg in client.list_aggs(
            ticker=ticker, from_=start_date, to=todays_date, multiplier=1, timespan="day"
        ):
            daily_ticker_data.append(ticker_agg)

        df = pd.DataFrame(daily_ticker_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"].dt.date
        return df

    def pull_all_stock_data(ticker_list: list[str], start_date: str, sleep_time: int = 30) -> dict:
        stock_data_dict = {}
        for ticker in ticker_list:
            stock_data_dict[ticker] = pull_stock_data(ticker, start_date)
            time.sleep(sleep_time)
        return stock_data_dict

    ndx_df = pull_stock_data(ndx_ticker, start_date)
    df_dict = pull_all_stock_data(stock_list, start_date)
    df_dict[ndx_ticker] = ndx_df

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ticker_data.pkl"), "wb") as output_file:
        pickle.dump(df_dict, output_file)

@dsl.component
def process_data(input_dir: str, output_dir: str):
    import os
    import sys
    import pandas as pd
    import subprocess
    import pickle

    print('processing data...')
    with open(os.path.join(input_dir, "ticker_data.pkl"), "rb") as input_file:
        df_dict = pickle.load(input_file)

    ndx_ticker = "I:NDX"
    columns_to_process = ["open", "low", "high"]
    n_lags, max_window_size = 5, 5

    def generate_lag_features(df: pd.DataFrame, column: str, n_lags: int = 5) -> None:
        for t in range(1, n_lags + 1):
            df[f"{column}_lag{t}"] = df[f"{column}"].shift(t)

    def calculate_sliding_window_averages(df: pd.DataFrame, field_name: str, max_window_size: int = 5) -> None:
        for window_size in range(1, max_window_size + 1):
            for i in range(1, max_window_size - window_size + 1):
                column_name = f"{field_name}_window_avg_from_{i}_to_{i+window_size}"
                columns_to_window = [f"{field_name}_lag{c}" for c in range(i, i + window_size + 1)]
                df[column_name] = df[columns_to_window].mean(axis=1)

    def run_feature_pipeline(df: pd.DataFrame, features: list[str], n_lags: int = 5, max_window_size: int = 5) -> None:
        for c in features:
            generate_lag_features(df, c, n_lags)
            calculate_sliding_window_averages(df, c, max_window_size)

    def create_model_dataset(df: pd.DataFrame, main_ticker: str, columns_to_process: list[str], ticker_df_dict: dict, n_lags: int, max_window_size: int) -> pd.DataFrame:
        finaldf = df.copy()
        run_feature_pipeline(df, columns_to_process, n_lags, max_window_size)
        finaldf.columns = [f"{c}_{main_ticker.lower()}" for c in finaldf.columns]

        for ticker in ticker_df_dict:
            if ticker != main_ticker:
                ticker_df = ticker_df_dict[ticker].copy()
                run_feature_pipeline(ticker_df, columns_to_process, n_lags, max_window_size)
                finaldf = finaldf.merge(
                    ticker_df.rename(columns={c: f"{c}_{ticker.lower()}" for c in ticker_df.columns}),
                    how="left",
                    left_on=f"date_{main_ticker.lower()}",
                    right_on=f"date_{ticker.lower()}",
                )

        finaldf.drop_duplicates(subset=[f"date_{main_ticker.lower()}"], inplace=True)
        return finaldf

    ndx_df = df_dict[ndx_ticker]
    finaldf = create_model_dataset(ndx_df, ndx_ticker, columns_to_process, df_dict, n_lags, max_window_size)

    os.makedirs(output_dir, exist_ok=True)
    finaldf.to_parquet(os.path.join(output_dir, "processed_data.parquet"), index=False)

@dsl.component
def train_model(input_dir: str, output_dir: str):
    import sys
    import subprocess
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    import pandas as pd
    import numpy as np

    print('training model...')
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

    df = pd.read_parquet(os.path.join(input_dir, "processed_data.parquet"))

    feature_list = sorted([c for c in df.columns if ("lag_" in c or "window_" in c) and "ndx" not in c])
    features = torch.from_numpy(df[feature_list].values[5:, :].astype(np.float32))
    labels = torch.from_numpy(df["open_i:ndx"].values[5:].astype(np.float32))

    input_size = features.shape[1]
    hidden_size = 32
    output_size = 1
    model = SimpleNN(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    num_epochs = 200
    for epoch in range(num_epochs):
        outputs = model(features)
        loss = criterion(outputs, labels.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

@dsl.component
def make_predictions(model_dir: str, data_dir: str, output_dir: str):
    import sys
    import subprocess
    import torch
    import torch.nn as nn
    import os
    import pandas as pd
    import numpy as np
    import datetime

    print('running predictions on model...')
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

    df = pd.read_parquet(os.path.join(data_dir, "processed_data.parquet"))

    feature_list = sorted([c for c in df.columns if ("lag_" in c or "window_" in c) and "ndx" not in c])
    features = torch.from_numpy(df[feature_list].values[5:, :].astype(np.float32))

    input_size = features.shape[1]
    hidden_size = 32
    output_size = 1
    model = SimpleNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    model.eval()

    with torch.no_grad():
        predictions = model(features)

    df.loc[5:, "predictions"] = predictions.numpy()
    df["run_date"] = datetime.datetime.now().date().strftime("%Y-%m-%d")

    os.makedirs(output_dir, exist_ok=True)
    dfss = df[["date_i:ndx", "open_i:ndx", "predictions", "run_date"]]
    dfss.to_parquet(
        os.path.join(output_dir, "predictions.parquet"), index=False
    )
    print(f"predictions = {dfss.tail()}")

# Create necessary directories
os.makedirs('./archive', exist_ok=True)
os.makedirs('./data', exist_ok=True)
os.makedirs('./predictions', exist_ok=True)

# Define the pipeline
@dsl.pipeline(
    name='Stock Data ELT Pipeline',
    description='Extract, Load, and Transform stock data, then train a model and make predictions.'
)
def stock_data_pipeline(api_key: str):
    # Use local directories
    archive_dir = './archive'
    data_dir = './data'
    predictions_dir = './predictions'

    # Define pipeline steps
    #fetch_op = fetch_stock_data(api_key=api_key, output_dir=archive_dir)
    process_op = process_data(input_dir=archive_dir, output_dir=data_dir)
    train_op = train_model(input_dir=data_dir, output_dir=data_dir)
    predict_op = make_predictions(model_dir=data_dir, data_dir=data_dir, output_dir=predictions_dir)

    # Set up dependencies
    #process_op.after(fetch_op)
    train_op.after(process_op)
    predict_op.after(train_op)

# Compile and run the pipeline
if __name__ == '__main__':

    from kfp.compiler import Compiler
    Compiler().compile(stock_data_pipeline, 'stock_data_pipeline.yaml')

    # Execute the pipeline directly
    stock_data_pipeline(api_key=os.environ['POLYGON_API_KEY'])

