# clean_options.py

import pandas as pd
import os
from datetime import datetime


def load_data(file_path):
    """Load CSV file if it exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean and enrich the options data."""
    df = df.dropna(subset=["bid", "ask", "openInterest"]).copy()
    df["T"] = df["remaining"] / 365
    df["midPrice"] = (df["bid"] + df["ask"]) / 2

    # Filter out options with zero implied volatility
    df = df[df["impliedVolatility"] > 0].copy()

    return df


def main():
    today_str = datetime.now().strftime("%Y_%m_%d")

    # Definir rutas absolutas respecto al script
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_dir, "data", "raw")
    processed_data_path = os.path.join(project_dir, "data", "processed")

    calls_path = os.path.join(raw_data_path, f"sp500_calls_{today_str}.csv")
    puts_path = os.path.join(raw_data_path, f"sp500_puts_{today_str}.csv")

    print("[INFO] Loading raw data...")
    df_calls = load_data(calls_path)
    df_puts = load_data(puts_path)

    print("[INFO] Cleaning and enriching data...")
    df_calls = clean_data(df_calls)
    df_puts = clean_data(df_puts)

    os.makedirs(processed_data_path, exist_ok=True)
    cleaned_calls_path = os.path.join(processed_data_path, f"calls_{today_str}.csv")
    cleaned_puts_path = os.path.join(processed_data_path, f"puts_{today_str}.csv")

    print("[INFO] Saving cleaned data...")
    df_calls.to_csv(cleaned_calls_path, index=False)
    df_puts.to_csv(cleaned_puts_path, index=False)

    print(f"[DONE] Cleaned files saved to: {processed_data_path}")


if __name__ == "__main__":
    main()
