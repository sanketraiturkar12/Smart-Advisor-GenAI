import yfinance as yf
import pandas as pd
import mysql.connector
import pymysql

# Fetch the list of S&P 500 companies
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

# Define the period for historical data
period = "60d"

# Initialize an empty DataFrame to store all stock data
all_stock_data = pd.DataFrame()

# Loop through each ticker and fetch the stock data
for ticker in sp500_tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist["Ticker"] = ticker
    hist.reset_index(inplace=True)
    hist = hist[["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]]
    
    # Check if hist is not empty before concatenating
    if not hist.empty:
        all_stock_data = pd.concat([all_stock_data, hist], ignore_index=True)

# Connect to MySQL database
MYSQL_CONFIG = {
    "host": "smartassistsql.mysql.database.azure.com",
    "user": "zensar",
    "password": "Admin@123",
    "database": "stock_sentiment_db",
    "ssl": {
        "ca": "C:/Users/SR76875/Downloads/DigiCertGlobalRootCA.crt.pem"  # Update this path to the CA certificate
    }
}

try:
    connection = pymysql.connect(**MYSQL_CONFIG)
    print("Connection successful!")
    connection.close()
except pymysql.MySQLError as e:
    print(f"Error: {e}")

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the MySQL database."""
    return pymysql.connect(**MYSQL_CONFIG)


# Insert data into the table

conn = get_db_connection()
cursor = conn.cursor()

# Prepare the insert query
insert_query = """
INSERT INTO sp500_stock_data (Ticker, Date, Open, High, Low, Close, Volume)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

# Convert DataFrame rows to a list of tuples for insertion
data_to_insert = all_stock_data.values.tolist()

# Use executemany to insert multiple rows at once
cursor.executemany(insert_query, data_to_insert)

# Commit the transaction
conn.commit()

# Print confirmation message
print(f"Stock data for S&P 500 saved to MySQL database.")

# Close the database connection
cursor.close()
conn.close()

