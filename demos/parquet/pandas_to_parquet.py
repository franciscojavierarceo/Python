import os
import pandas as pd


def write_sample_csv():
    d = {
        "truck": "Ford",
        "description": {
            "features": {
                "steering": "4 wheel drive",
                "engine": "8-cylinders",
                "climate": "Air Conditioning",
            },
        },
        "year": 2022,
        "cost": 42341.99,
    }
    df = pd.DataFrame(d)
    df.to_csv("example.csv", index=False)
    print("...csv written")


def main():
    if "example.csv" not in os.listdir("./"):
        print("writing csv...")
        write_sample_csv()

    df = pd.read_csv("example.csv")

    print("writing parquet file...")
    df.to_parquet("example.parquet.gzip", compression="gzip")
    print("...parquet file written")


if __name__ == "__main__":
    main()
