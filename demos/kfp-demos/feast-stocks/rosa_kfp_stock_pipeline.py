from kfp import dsl
from kfp.local import init, SubprocessRunner
import os
import sys
import subprocess

base_image = 'python:3.10'

my_requirements = """
aiohttp==3.9.5
aiohttp-cors==0.7.0
aiosignal==1.3.1
annotated-types==0.7.0
anyio==4.4.0
appnope==0.1.4
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
arrow==1.3.0
asttokens==2.4.1
async-lru==2.0.4
async-timeout==4.0.3
attrs==23.2.0
Babel==2.15.0
beautifulsoup4==4.12.3
black==23.3.0
bleach==6.1.0
cachetools==5.4.0
certifi==2024.7.4
cffi==1.16.0
charset-normalizer==3.3.2
click==8.1.7
cloudevents==1.11.0
cloudpickle==3.0.0
colorama==0.4.6
colorful==0.5.6
comm==0.2.2
contourpy==1.2.1
cycler==0.12.1
dask==2024.7.0
dask-expr==1.1.7
debugpy==1.8.2
decorator==5.1.1
defusedxml==0.7.1
deprecation==2.1.0
dill==0.3.8
distlib==0.3.8
dnspython==2.6.1
docstring_parser==0.16
email_validator==2.2.0
exceptiongroup==1.2.2
executing==2.0.1
fastapi==0.109.2
fastapi-cli==0.0.4
fastjsonschema==2.20.0
filelock==3.15.4
fonttools==4.53.1
fqdn==1.5.1
frozenlist==1.4.1
fsspec==2024.6.1
google-api-core==2.19.1
google-auth==2.32.0
google-cloud-core==2.4.1
google-cloud-storage==2.17.0
google-crc32c==1.5.0
google-resumable-media==2.7.1
googleapis-common-protos==1.63.2
grpcio==1.64.1
gunicorn==22.0.0
h11==0.14.0
httpcore==1.0.5
httptools==0.6.1
httpx==0.26.0
idna==3.7
importlib_metadata==8.0.0
ipykernel==6.29.5
ipython==8.26.0
ipywidgets==8.1.3
isoduration==20.11.0
jedi==0.19.1
Jinja2==3.1.4
json5==0.9.25
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2023.12.1
jupyter==1.0.0
jupyter-console==6.6.3
jupyter-events==0.10.0
jupyter-lsp==2.2.5
jupyter_client==8.6.2
jupyter_core==5.7.2
jupyter_server==2.14.2
jupyter_server_terminals==0.5.3
jupyterlab==4.2.3
jupyterlab_pygments==0.3.0
jupyterlab_server==2.27.2
jupyterlab_widgets==3.0.11
kfp==2.8.0
kfp-pipeline-spec==0.3.0
kfp-server-api==2.0.5
kiwisolver==1.4.5
kubernetes==26.1.0
locket==1.0.0
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.1
matplotlib-inline==0.1.7
mdurl==0.1.2
mistune==3.0.2
mmh3==4.1.0
mpmath==1.3.0
msgpack==1.0.8
multidict==6.0.5
mypy==1.10.1
mypy-extensions==1.0.0
mypy-protobuf==3.6.0
nbclient==0.10.0
nbconvert==7.16.4
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.3
notebook==7.2.1
notebook_shim==0.2.4
numpy==1.26.4
oauthlib==3.2.2
opencensus==0.11.4
opencensus-context==0.1.3
orjson==3.10.6
overrides==7.7.0
packaging==24.1
pandas==2.2.2
pandocfilters==1.5.1
parso==0.8.4
partd==1.4.2
pathspec==0.12.1
pexpect==4.9.0
pillow==10.4.0
platformdirs==4.2.2
polygon-api-client==1.14.2
prometheus_client==0.20.0
prompt_toolkit==3.0.47
proto-plus==1.24.0
protobuf==4.25.3
psutil==5.9.8
ptyprocess==0.7.0
pure-eval==0.2.2
py-spy==0.3.14
pyarrow==17.0.0
pyasn1==0.6.0
pyasn1_modules==0.4.0
pycparser==2.22
pydantic==2.8.2
pydantic_core==2.20.1
Pygments==2.18.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
python-json-logger==2.0.7
python-multipart==0.0.9
pytz==2024.1
PyYAML==6.0.1
pyzmq==26.0.3
qtconsole==5.5.2
QtPy==2.4.1
ray==2.10.0
referencing==0.35.1
requests==2.32.3
requests-oauthlib==2.0.0
requests-toolbelt==0.10.1
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==13.7.1
rpds-py==0.19.0
rsa==4.9
Send2Trash==1.8.3
shellingham==1.5.4
six==1.16.0
smart-open==7.0.4
sniffio==1.3.1
soupsieve==2.5
SQLAlchemy==2.0.31
stack-data==0.6.3
starlette==0.36.3
sympy==1.13.0
tabulate==0.9.0
tenacity==8.5.0
terminado==0.18.1
timing-asgi==0.3.1
tinycss2==1.3.0
toml==0.10.2
tomli==2.0.1
toolz==0.12.1
torch==2.3.1
tornado==6.4.1
tqdm==4.66.4
traitlets==5.14.3
typeguard==4.3.0
typer==0.12.3
types-protobuf==5.27.0.20240626
types-python-dateutil==2.9.0.20240316
typing_extensions==4.12.2
tzdata==2024.1
uri-template==1.3.0
urllib3==1.26.19
uvicorn==0.21.1
uvloop==0.19.0
virtualenv==20.26.3
watchfiles==0.22.0
wcwidth==0.2.13
webcolors==24.6.0
webencodings==0.5.1
websocket-client==1.8.0
websockets==12.0
widgetsnbextension==4.0.11
wrapt==1.16.0
yarl==1.9.4
zipp==3.19.2
""".split("\n")[1:-1]

@dsl.component(base_image=base_image, packages_to_install=my_requirements)
def fetch_stock_data(api_key: str, output_dir: str) -> None:
    import sys
    import subprocess
    import time
    import datetime
    import os
    import pickle
    import pandas as pd
    from polygon import RESTClient

    print("fetching stock data...")
    ndx_ticker = "I:NDX"
    start_date = "2024-01-01"
    output_filename = "ticker_data.pkl"
    log_file = "successful_dates.log"
    todays_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
    client = RESTClient(api_key)
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
        business_days = (
            pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
        )
        return [date for date in business_days if date not in successful_dates]

    def pull_stock_data(ticker: str, start_date: str) -> pd.DataFrame:
        daily_ticker_data = []
        for ticker_agg in client.list_aggs(
            ticker=ticker,
            from_=start_date,
            to=todays_date,
            multiplier=1,
            timespan="day",
        ):
            daily_ticker_data.append(ticker_agg)

        df = pd.DataFrame(daily_ticker_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"].dt.date
        return df

    def pull_all_stock_data(
        ticker_list: list[str], start_date: str, sleep_time: int = 30
    ) -> dict:
        stock_data_dict = {}
        for ticker in ticker_list:
            stock_data_dict[ticker] = pull_stock_data(ticker, start_date)
            time.sleep(sleep_time)
        return stock_data_dict

    def get_or_load_historical_data(
        output_dir: str,
        output_filename: str,
        start_date: str,
    ) -> Tuple[pd.DataFrame, dict]:
        if output_filename in os.listdir(output_dir):
            print("loading stored data...")
            with open(os.path.join(output_dir, output_filename), "rb") as output_file:
                df_dict = pickle.load(output_file)
                ndx_df = df_dict[ndx_ticker]
        else:
            print("no stored data found, calling polygon api...")
            ndx_df = pull_stock_data(ndx_ticker, start_date)
            df_dict = pull_all_stock_data(stock_list, start_date)
            df_dict[ndx_ticker] = ndx_df

            with open(os.path.join(output_dir, output_filename), "wb") as output_file:
                pickle.dump(df_dict, output_file)
        return ndx_df, df_dict

    successful_dates = get_successful_dates(log_file)
    dates_to_pull = get_dates_to_pull(start_date, todays_date, successful_dates)

    ndx_df, df_dict = get_or_load_historical_data(
        output_dir,
        output_filename,
        start_date,
    )


@dsl.component(base_image=base_image, packages_to_install=my_requirements)
def process_data(input_dir: str, output_dir: str) -> None:
    import os
    import sys
    import pandas as pd
    import subprocess
    import pickle

    print("processing data...")
    with open(os.path.join(input_dir, "ticker_data.pkl"), "rb") as input_file:
        df_dict = pickle.load(input_file)

    ndx_ticker = "I:NDX"
    columns_to_process = ["open", "low", "high"]
    n_lags, max_window_size = 5, 5

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
        main_ticker: str,
        columns_to_process: list[str],
        ticker_df_dict: dict,
        n_lags: int,
        max_window_size: int,
    ) -> pd.DataFrame:
        finaldf = df.copy()
        run_feature_pipeline(df, columns_to_process, n_lags, max_window_size)
        finaldf.columns = [f"{c}_{main_ticker.lower()}" for c in finaldf.columns]

        for ticker in ticker_df_dict:
            if ticker != main_ticker:
                ticker_df = ticker_df_dict[ticker].copy()
                run_feature_pipeline(
                    ticker_df, columns_to_process, n_lags, max_window_size
                )
                finaldf = finaldf.merge(
                    ticker_df.rename(
                        columns={c: f"{c}_{ticker.lower()}" for c in ticker_df.columns}
                    ),
                    how="left",
                    left_on=f"date_{main_ticker.lower()}",
                    right_on=f"date_{ticker.lower()}",
                )

        finaldf.drop_duplicates(subset=[f"date_{main_ticker.lower()}"], inplace=True)
        return finaldf

    def save_data_to_parquet(df: pd.DataFrame, base_path: str):
        unique_dates = df["date_i:ndx"].astype(str).unique().tolist()
        missing_dates = [
            j for j in unique_dates if f"date_i:ndx={j}" not in os.listdir(base_path)
        ]
        if len(missing_dates) > 0:
            print(f"exporting {len(missing_dates)} more file(s)")
            dfss = df[
                df["date_i:ndx"].astype(str).str.contains("|".join(missing_dates))
            ]
            dfss.to_parquet(base_path, index=False, partition_cols=["date_i:ndx"])
        else:
            print("exporting 0 files - no new data found")

    ndx_df = df_dict[ndx_ticker]
    finaldf = create_model_dataset(
        ndx_df, ndx_ticker, columns_to_process, df_dict, n_lags, max_window_size
    )

    os.makedirs(output_dir, exist_ok=True)
    save_data_to_parquet(finaldf, os.path.join(output_dir))
    print(f"{finaldf.shape[0]} records in dataset...")

@dsl.component(base_image=base_image, packages_to_install=my_requirements)
def train_model(input_dir: str, output_dir: str) -> None:
    import sys
    import subprocess
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    import pandas as pd
    import numpy as np

    print("training model...")

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

    feature_list = sorted(
        [c for c in df.columns if ("lag_" in c or "window_" in c) and "ndx" not in c]
    )
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

@dsl.component(base_image=base_image, packages_to_install=my_requirements)
def make_predictions(model_dir: str, data_dir: str, output_dir: str) -> None:
    import sys
    import subprocess
    import torch
    import torch.nn as nn
    import os
    import pandas as pd
    import numpy as np
    import datetime

    print("running predictions on model...")

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

    feature_list = sorted(
        [c for c in df.columns if ("lag_" in c or "window_" in c) and "ndx" not in c]
    )
    features = torch.from_numpy(df[feature_list].values[5:, :].astype(np.float32))
    labels = torch.from_numpy(df["open_i:ndx"].values[5:].astype(np.float32))

    input_size = features.shape[1]
    hidden_size = 32
    output_size = 1
    model = SimpleNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    model.eval()

    with torch.no_grad():
        predictions = model(features)

    print(f"RMSE = {torch.sqrt( torch.mean( (predictions- labels) ** 2) )}")
    df.loc[5:, "predictions"] = predictions.numpy()
    df["run_date"] = datetime.datetime.now().date().strftime("%Y-%m-%d")

    os.makedirs(output_dir, exist_ok=True)
    dfss = df[["date_i:ndx", "open_i:ndx", "predictions", "run_date"]]
    dfss.to_parquet(os.path.join(output_dir, "predictions.parquet"), index=False)
    print(f"predictions = {dfss.tail()}")


#@dsl.component#(packages_to_install=my_requirements)
#def materialize_online_store(model_dir: str, data_dir: str, output_dir: str) -> None: #    import feast
#
#    print(feast.__version__)


# Create necessary directories
os.makedirs("./archive", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./predictions", exist_ok=True)


# Define the pipeline
@dsl.pipeline(
    name="Stock Data ELT Pipeline",
    description="Extract, Load, and Transform stock data, then train a model and make predictions.",
)
def stock_data_pipeline(api_key: str) -> None:
    # Use local directories
    archive_dir = "./archive"
    data_dir = "./data"
    predictions_dir = "./predictions"

    # Define pipeline steps
    fetch_op = fetch_stock_data(api_key=api_key, output_dir=archive_dir)
    process_op = process_data(input_dir=archive_dir, output_dir=data_dir)
    train_op = train_model(input_dir=data_dir, output_dir=data_dir)
    predict_op = make_predictions(
        model_dir=data_dir, data_dir=data_dir, output_dir=predictions_dir
    )
    #tmp = materialize_online_store(
    #    model_dir=data_dir, data_dir=data_dir, output_dir=predictions_dir
    #)

    # Set up dependencies
    process_op.after(fetch_op)
    train_op.after(process_op)
    predict_op.after(train_op)
    return None


# Compile the pipeline
if __name__ == "__main__":
    from kfp.compiler import Compiler

    kfp_output_yaml = "stock_data_pipeline.yaml"
    Compiler().compile(stock_data_pipeline, kfp_output_yaml)
    print(f"kfp compiled to {kfp_output_yaml}")

