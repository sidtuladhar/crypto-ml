import os
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


def extract_stock_data(url):
    """
    Extracts daily stock data from Yahoo Finance.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    table_class = "table yf-j5d1ld noDl"
    row_class = "yf-j5d1ld"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", class_=table_class)
        if not table:
            raise ValueError("Table not found!")

        # Extract rows from the table
        rows = table.find_all("tr", class_=row_class)
        if not rows:
            raise ValueError("No rows found in the table!")

        # Extract data from each row
        data = []
        for row in rows:
            cols = row.find_all("td", class_=row_class)
            if len(cols) == 7:  # Ensure the row has the expected number of columns
                row_data = [col.text.strip() for col in cols]
                data.append(row_data)

        # Create a DataFrame from the scraped data
        df = pd.DataFrame(
            data,
            columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
        )
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def save_to_csv(df, filename):
    """
    Saves a DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): The name of the CSV file.
    """
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)

    print(f"Data saved to {csv_path}")


def generate_csv():
    apple_url = "https://finance.yahoo.com/quote/AAPL/history/?period1=1577553804&period2=1735406601"
    tesla_url = "https://finance.yahoo.com/quote/TSLA/history/?period1=1577553785&period2=1735406583"
    jp_morgan_url = "https://finance.yahoo.com/quote/JPM/history/?period1=1577553765&period2=1735406563"
    shopify_url = "https://finance.yahoo.com/quote/SHOP/history/?period1=1577553744&period2=1735406541"
    nvidia_url = "https://finance.yahoo.com/quote/NVDA/history/?period1=1577553555&period2=1735406352"

    apple_df = extract_stock_data(apple_url)
    tesla_df = extract_stock_data(tesla_url)
    jp_morgan_df = extract_stock_data(jp_morgan_url)
    shopify_df = extract_stock_data(shopify_url)
    nvidia_df = extract_stock_data(nvidia_url)

    save_to_csv(apple_df, "apple_data.csv")
    save_to_csv(tesla_df, "tesla_data.csv")
    save_to_csv(jp_morgan_df, "jp_morgan_data.csv")
    save_to_csv(shopify_df, "shopify_data.csv")
    save_to_csv(nvidia_df, "nvidia_data.csv")


def load_csv(file_path):
    file_path = Path(file_path)  # Ensure it's a Path object

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


if __name__ == "__main__":
    generate_csv()
    apple_df = load_csv("./data/apple_data.csv")
    tesla_df = load_csv("./data/tesla_data.csv")
    jp_morgan_df = load_csv("./data/jp_morgan_data.csv")
    shopify_df = load_csv("./data/shopify_data.csv")
    nvidia_df = load_csv("./data/nvidia_data.csv")

    print(apple_df.head())
