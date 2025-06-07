import pandas as pd
import yfinance as yf
import requests
import os

from datetime import datetime
from bs4 import BeautifulSoup


def get_tickers():
    """Returns the tickers for all the S&P500 companies using the Wikipedia page
    Outputs:
        tickers - list of tickers for every company in the S&P500
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("table")  # tickers are contained in a table
    tickers = []
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if cols:
            tickers.append(cols[0].text.strip())
    return tickers


def convert_options_to_dict(df, expiration_date, underlying_price):
    """Converts a DataFrame of options data into a list of dictionaries with specific fields.
    Args:
        df (pd.DataFrame): DataFrame containing options data.
        expiration_date (str): Expiration date in 'YYYY-MM-DD' format.
        underlying_price (float): Current price of the underlying asset.
        Returns:
        list: List of dictionaries with selected fields from the DataFrame.
    """
    expiration_datetime = datetime.strptime(expiration_date, "%Y-%m-%d")
    df = df.assign(expiration=expiration_datetime, price=underlying_price)
    df["lastTradeDate"] = df["lastTradeDate"].apply(lambda x: x.replace(tzinfo=None))

    return df.to_dict(orient="records")


def get_option_data(ticker):
    """Returns the options data for a given ticker
    Inputs:
        ticker - string, the ticker symbol of the stock
    Outputs:
        calls - list of dictionaries containing call options data
        puts - list of dictionaries containing put options data
    """
    calls = []
    puts = []

    stock = yf.Ticker(ticker)
    for expiration_date in stock.options:
        opt = stock.option_chain(expiration_date)
        opt_calls = opt.calls
        opt_puts = opt.puts
        underlying_price = opt.underlying["regularMarketPrice"]
        calls.append(
            convert_options_to_dict(opt_calls, expiration_date, underlying_price)
        )
        puts.append(
            convert_options_to_dict(opt_puts, expiration_date, underlying_price)
        )

    return calls, puts


def generate_options_df(options):
    """
    Generates a dataframe from the options data in the format returned by get_options_data.
    """
    options = [item for sublist in options for item in sublist]
    df_opts = pd.DataFrame.from_records(options)
    df_opts["duration"] = df_opts["expiration"] - df_opts["lastTradeDate"]
    df_opts["duration"] = df_opts["duration"].apply(lambda x: x.days)
    df_opts["remaining"] = df_opts["expiration"].apply(
        lambda x: (x - datetime.now()).days
    )

    return df_opts


def get_all_options_data(tickers):
    """Fetches options data for a list of tickers.
    Args:
        tickers (list): List of ticker symbols.
    Returns:
        tuple: Two DataFrames, one for call options and one for put options.
    """

    # with ThreadPoolExecutor(max_workers=100) as p:
    #      results = p.map(get_options_data, tickers)

    results = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        try:
            result = get_option_data(ticker)
            results.append(result)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    calls = []
    puts = []

    for result in results:
        calls.extend(result[0])
        puts.extend(result[1])

    df_calls = generate_options_df(calls)
    df_puts = generate_options_df(puts)

    return df_calls, df_puts


def main():
    today_str = datetime.now().strftime("%Y_%m_%d")

    # Path to the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    print("[INFO] Fetching S&P 500 tickers...")
    tickers = get_tickers()

    print("[INFO] Scraping options data...")
    df_calls, df_puts = get_all_options_data(tickers)

    print("[INFO] Saving CSV files...")
    df_calls.to_csv(os.path.join(data_dir, f"sp500_calls_{today_str}.csv"), index=False)
    df_puts.to_csv(os.path.join(data_dir, f"sp500_puts_{today_str}.csv"), index=False)

    print("[DONE] Files saved.")


if __name__ == "__main__":
    main()
