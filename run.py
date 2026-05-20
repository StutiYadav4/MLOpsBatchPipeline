import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def write_metrics(output_path, metrics):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_config(config_path):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = ["seed", "window", "version"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing config field: {key}")

    return config

def load_dataset(input_path):
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        # Read raw file
        with open(input_path, "r") as f:
            lines = [line.strip().strip('"') for line in f.readlines()]

        if len(lines) == 0:
            raise ValueError("Input CSV is empty")

        # Split header and rows manually
        headers = lines[0].split(",")

        data = [row.split(",") for row in lines[1:]]

        # Create dataframe
        df = pd.DataFrame(data, columns=headers)

        # Clean columns
        df.columns = df.columns.str.strip().str.lower()

        # Convert close column to numeric
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

    except Exception as e:
        raise ValueError(f"Invalid CSV format: {str(e)}")

    if df.empty:
        raise ValueError("Input CSV is empty")

    if "close" not in df.columns:
        raise ValueError("Required column 'close' missing")

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-file", required=True)

    args = parser.parse_args()

    setup_logger(args.log_file)

    start_time = time.time()

    try:
        logging.info("Job started")

        # Load config
        config = load_config(args.config)

        seed = config["seed"]
        window = config["window"]
        version = config["version"]

        np.random.seed(seed)

        logging.info(
            f"Config loaded successfully | seed={seed}, "
            f"window={window}, version={version}"
        )

        # Load dataset
        df = load_dataset(args.input)

        logging.info(f"Rows loaded: {len(df)}")

        # Rolling mean
        logging.info("Computing rolling mean")

        df["rolling_mean"] = (
            df["close"]
            .rolling(window=window)
            .mean()
        )

        # Signal generation
        logging.info("Generating signals")

        df["signal"] = np.where(
            df["close"] > df["rolling_mean"],
            1,
            0
        )

        signal_rate = float(df["signal"].mean())

        latency_ms = int((time.time() - start_time) * 1000)

        metrics = {
            "version": version,
            "rows_processed": int(len(df)),
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success"
        }

        write_metrics(args.output, metrics)

        logging.info(f"Metrics summary: {metrics}")
        logging.info("Job completed successfully")

        print(json.dumps(metrics, indent=2))

        sys.exit(0)

    except Exception as e:

        latency_ms = int((time.time() - start_time) * 1000)

        error_metrics = {
            "version": "v1",
            "status": "error",
            "error_message": str(e)
        }

        write_metrics(args.output, error_metrics)

        logging.exception("Pipeline failed")

        print(json.dumps(error_metrics, indent=2))

        sys.exit(1)


if __name__ == "__main__":
    main()