from flask import Flask, jsonify, request
from flask_cors import CORS
import pymysql
from sqlalchemy import create_engine
import pandas as pd
from openai import AzureOpenAI
import matplotlib.pyplot as plt
import base64
import io
from datetime import datetime, timedelta
import pytz
import matplotlib
import os
import mplfinance as mpf
import mysql.connector
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor








matplotlib.use('Agg')





# --- Configuration ---
client = AzureOpenAI(
    
    api_key="m9QfxXYxdW8W0KUVxXpBkmBvzgb0cPcLf6uQm81MKzaVw7lUIGl0JQQJ99AJACYeBjFXJ3w3AAABACOGwjQW",
    api_version="2025-01-01-preview",
    azure_endpoint="https://opneai-gpt-4o.openai.azure.com",
    azure_deployment="gpt-4" 
)

MYSQL_CONFIG = {
    "host": "smartassistsql.mysql.database.azure.com",
    "user": "zensar",
    "password": "Admin@123",
    "database": "stock_sentiment_db",
    "ssl": {
        "ca": "C:/Users/SR76875/Downloads/DigiCertGlobalRootCA.crt.pem"  # Ensure this path is correct
    }
}

ADDITIONAL_DB_CONFIG = {
    "host": "smartassistsql.mysql.database.azure.com",
    "user": "zensar",
    "password": "Admin@123",
    "database": "sa_test",
    "ssl": {
        "ca": "C:/Users/SR76875/Downloads/DigiCertGlobalRootCA.crt.pem"  # Ensure this path is correct
    }
}

# MYSQL_CONFIG = {
#     "host": "localhost",  # Change to your local MySQL server if needed
#     "user": "root",
#     "password": "root",
#     "database": "stock_sentiment_db"
# }

# ADDITIONAL_DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "root",
#     "database": "sa_test"
# }


app = Flask(__name__)

CORS(app)

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the MySQL database."""
    return pymysql.connect(**MYSQL_CONFIG)


# def get_db_connection():
#     """Establishes a secure connection to the MySQL database using mysql.connector."""
#     try:
#         conn = mysql.connector.connect(
#             host="smartassistsql.mysql.database.azure.com",
#             user="zensar",
#             password="Admin@123",
#             database="stock_sentiment_db",
#             ssl_ca="C:/Users/SR76875/Downloads/DigiCertGlobalRootCA.crt.pem"
#         )
#         print("Database connection successful!")
#         return conn
#     except mysql.connector.Error as err:
#         print(f"Error: {err}")
#         raise

def get_additional_db_connection():
    """Establishes a connection to the additional MySQL database."""
    return pymysql.connect(**ADDITIONAL_DB_CONFIG)


@app.route('/api/stock_holdings', methods=['GET'])
def get_stock_holdings():
    """Fetches stock holdings data for a client and calculates portfolio sentiment score."""
    try:
        # Extract query parameters
        client_id = request.args.get('client_id', type=int)

        # Validate required parameters
        if not client_id:
            return jsonify({"error": "client_id is a required parameter"}), 400

        # SQL query to fetch stock holdings and sentiment scores
        qry = f"""
        SELECT  
            security.ticker,
            client.ClientID,
            security.SecurityID,
            position.`Market Value` AS marketValue,
            risk_profile.Risk_Profile_Desc,
            ROUND(SUM(sentiment.sentiment_score * position.quantity * quote.LastPrice) / SUM(position.quantity * quote.LastPrice), 2) AS Sentiment_score,
            TRUNCATE((quote.priceChangePct * 100), 2) AS priceChangePct,
            FORMAT(quote.LastPrice, 2) AS lastPrice,
            position.Quantity AS quantity,
            risk_profile.risk_profile_name,
            currency.currency_symbol,
            '%' AS changeType,
            IFNULL(ROUND(SUM(quote.esgScore * position.quantity * quote.LastPrice) / SUM(position.quantity * quote.LastPrice), 2), 10.21) AS esg_score
        FROM
            client
        JOIN position ON position.ClientID = client.ClientID
        JOIN currency ON position.CurrencyID = currency.CurrencyID
        JOIN sentiment ON sentiment.SE_Type_ref_ID = position.SecurityID
        JOIN sentiment_entity_type ON sentiment.SE_Type_Id = sentiment_entity_type.SE_Type_Id
        JOIN portfolio_action ON sentiment.Sentiment_score BETWEEN portfolio_action.Sentiment_Score_LB AND portfolio_action.Sentiment_Score_UB
        JOIN portfolio_action_type ON portfolio_action.ActionTypeID = portfolio_action_type.ActionTypeID
        JOIN security ON position.securityID = security.securityID
        JOIN risk_profile ON client.riskprofileID = risk_profile.riskprofileID
        JOIN quote ON position.securityID = quote.SecurityID
        WHERE
            sentiment_entity_type.se_type = 'Security'
            AND security.SecurityID NOT IN (101, 102, 103)
            AND sentiment.sentiment_date BETWEEN '2021-09-30 00:00:00' AND '2021-09-30 23:59:59'
            AND client.ClientID = {client_id}
            AND quote.DateTime BETWEEN '2021-05-25 00:00:00' AND '2021-05-25 23:59:59'
        GROUP BY
            security.ticker,
            client.ClientID,
            security.SecurityID,
            position.`Market Value`,
            risk_profile.Risk_Profile_Desc,
            position.Quantity,
            risk_profile.risk_profile_name,
            currency.currency_symbol,
            quote.priceChangePct,
            quote.LastPrice,
            portfolio_action.priority
        ORDER BY
            portfolio_action.priority ASC;
        """

        # Execute the query using the additional database connection
        connection = get_additional_db_connection()
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(qry)
            data = cursor.fetchall()

        # Calculate the portfolio sentiment score
        sentiment_scores = [row['Sentiment_score'] for row in data if row['Sentiment_score'] is not None]
        if sentiment_scores:
            portfolio_sentiment_score = round(sum(sentiment_scores) / len(sentiment_scores), 2)
        else:
            portfolio_sentiment_score = None

        # Add the portfolio sentiment score to the response
        response = {
            "client_id": client_id,
            "portfolio_sentiment_score": portfolio_sentiment_score,
            # "holdings": data
        }

        # Return the data as JSON
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

def load_stock_data(tickers=None):
    """Loads stock data from the MySQL database. Filters by tickers if provided."""
    try:
        # URL-encode the password
        encoded_password = quote_plus(MYSQL_CONFIG['password'])

        # Create a SQLAlchemy engine with the corrected connection string
        engine = create_engine(
            f"mysql+pymysql://{MYSQL_CONFIG['user']}:{encoded_password}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}?ssl_ca={MYSQL_CONFIG['ssl']['ca']}"
        )
        print(f"SQLAlchemy connection string: {engine.url}")
        
        # engine = create_engine(
        #     f"mysql+pymysql://{MYSQL_CONFIG['user']}:{encoded_password}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
        # )
        # print(f"SQLAlchemy connection string: {engine.url}")

        # Build the query
        if tickers:
            tickers_placeholder = ', '.join([f"'{ticker}'" for ticker in tickers])
            query = f"""
                SELECT Ticker, Date, Close, Volume 
                FROM sp500_stock_data 
                WHERE Ticker IN ({tickers_placeholder}) 
                AND Date >= CURDATE() - INTERVAL 30 DAY
            """
        else:
            query = """
                SELECT Ticker, Date, Close, Volume 
                FROM sp500_stock_data 
                WHERE Date >= CURDATE() - INTERVAL 30 DAY
            """

        # Query the database and load data into a DataFrame
        df = pd.read_sql(query, engine)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading stock data: {e}")
        raise
    

def generate_stock_chart(data, time_period="7d"):
    try:
        print("Starting chart generation...")
        # Ensure the 'Date' column is in datetime format
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        print("Date column converted to datetime.")

        # Remove rows with invalid dates
        data = data.dropna(subset=['Date'])
        print(f"Data after dropping invalid dates: {data.shape}")

        # Ensure the 'Date' column is timezone-aware
        if data['Date'].dt.tz is None:
            data['Date'] = data['Date'].dt.tz_localize(pytz.utc)
        print("Date column localized to UTC.")

        # Filter data based on the time period
        if time_period == "7d":
            start_date = datetime.now(pytz.utc) - timedelta(days=7)
        elif time_period == "1m":
            start_date = datetime.now(pytz.utc) - timedelta(days=30)
        else:
            raise ValueError("Invalid time_period. Use '7d' or '1m'.")

        filtered_data = data[data["Date"] >= start_date]
        print(f"Filtered data: {filtered_data.shape}")

        # Check if filtered data is empty
        if filtered_data.empty:
            raise ValueError(f"No data available for the specified time period: {time_period}")

        # Remove duplicates by keeping the last Close price for each Date
        filtered_data = filtered_data.sort_values('Date').drop_duplicates('Date', keep='last')
        print(f"Data after removing duplicates: {filtered_data.shape}")

        # Generate the graph
        plt.figure(figsize=(10, 5))
        plt.plot(filtered_data['Date'], filtered_data['Close'], marker='o', linestyle='-', color='b', label='Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title(f'Stock Price Trend ({time_period})')
        plt.grid(False)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the graph as a Base64-encoded image
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        print("Chart generation completed successfully.")
        return f"data:image/png;base64,{graph_url}"
    except Exception as e:
        print(f"Error during chart generation: {e}")
        raise

# --- Trend Calculation ---
def calculate_trend(data):
    """Calculates stock price trend as percentage change over time."""
    data["Price_Change"] = data["Close"].pct_change()
    data["Trend"] = data["Price_Change"].apply(
        lambda x: "Upward" if x > 0 else "Downward" if x < 0 else "Neutral"
    )
    return data

# --- Azure OpenAI Sentiment Analysis ---
def analyze_sentiment(trend, volume_change):
    """Uses Azure OpenAI to analyze stock trend and volume for sentiment."""
    prompt = f"""
    
    Based on the given stock market trend and trading volume change, analyze the sentiment and provide a recommendation. Format the output in bullet points.
    Trend: {trend}
    Volume Change: {volume_change}
    Output Format:

    Recommendation: [Buy/Sell/Hold]
    Details: [One-line justification based on trend and trading volume change]
    
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a financial sentiment analysis assistant."},
                  {"role": "user", "content": prompt}]
    )
    
    sentiment = response.choices[0].message.content
    print(sentiment)

    return sentiment.strip()




@app.route("/api/stock_sentiment", methods=["GET"])
def get_stock_sentiment():
    """Returns sentiment for a specific stock based on price trends and volume."""
    symbol = request.args.get("symbol")
    time_range = request.args.get("time_range", "7d")  # Default to 7 days if not provided

    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    # Load stock data and filter by symbol
    df = load_stock_data()
    print(df.head())  # Debug: Print the first few rows of the DataFrame

    df = df[df["Ticker"] == symbol]
    print(f"Filtered data for symbol {symbol}:\n{df}")  # Debug: Print filtered data for the symbol

    if df.empty:
        return jsonify({"error": "Stock symbol not found"}), 404

    # Filter data based on the time range
    if time_range == "7d":
        start_date = datetime.now(pytz.utc) - timedelta(days=7)
    elif time_range == "1m":
        start_date = datetime.now(pytz.utc) - timedelta(days=30)
    else:
        return jsonify({"error": "Invalid time_range. Use '7d' or '1m'."}), 400

    # Ensure the Date column is in datetime format and timezone-aware
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize("UTC")  # Convert to timezone-aware

    # Filter the DataFrame
    df = df[df["Date"] >= start_date]
    print(f"Data after filtering by time range ({time_range}):\n{df}")  # Debug: Print filtered data by time range

    # Check if the DataFrame is empty after filtering
    if df.empty:
        return jsonify({"error": f"No data available for {symbol} in the specified time range."}), 404

    # Calculate trends and get the latest data
    df = calculate_trend(df)
    if df.empty:
        return jsonify({"error": "No trend data available for the stock."}), 404

    latest_data = df.iloc[-1]  # Safely access the last row

    # Use ThreadPoolExecutor to run tasks in parallel
    with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        sentiment_future = executor.submit(analyze_sentiment, latest_data["Trend"], latest_data["Volume"])
        chart_future = executor.submit(generate_stock_chart, df, time_period=time_range)

        # Wait for both tasks to complete
        sentiment = sentiment_future.result()
        graph = chart_future.result()

    return jsonify({
        "symbol": symbol,
        "sentiment": sentiment,
        "chart": graph
    })
    # Store in MySQL
    # conn = get_db_connection()
    # cursor = conn.cursor()
    # cursor.execute(
    #     "INSERT INTO stock_sentiment (symbol, sentiment) VALUES (%s, %s)",
    #     (symbol, sentiment)
    # )
    # conn.commit()
    # conn.close()

    # return jsonify({"symbol": symbol, "sentiment": sentiment,"chart_7d": graph_7d,"chart_1m": graph_1m})

# --- API: Compare Stocks ---

def generate_comparison_chart(stock_data, time_period="7d"):
    """
    Generates a multi-line stock comparison chart for the specified time period.
    :param stock_data: Dictionary where keys are tickers and values are DataFrames containing stock data.
    :param time_period: "7d" for the past 7 days or "1m" for the last one month.
    :return: Base64-encoded image URL of the graph.
    """
    # Determine the start date based on the time period
    if time_period == "7d":
        start_date = datetime.now(pytz.utc) - timedelta(days=7)
    elif time_period == "1m":
        start_date = datetime.now(pytz.utc) - timedelta(days=30)
    else:
        raise ValueError("Invalid time_period. Use '7d' or '1m'.")

    plt.figure(figsize=(10, 6))  # Larger figure size for better readability

    # Filter data and plot for each ticker
    for ticker, data in stock_data.items():
        # Ensure the 'Date' column is in datetime format
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Ensure the 'Date' column is timezone-aware
        if data['Date'].dt.tz is None:
            data['Date'] = data['Date'].dt.tz_localize(pytz.utc)

        # Filter data based on the time period
        filtered_data = data[data["Date"] >= start_date]

        # Check if filtered data is empty
        if filtered_data.empty:
            raise ValueError(f"No data available for ticker {ticker} in the specified time period: {time_period}")

        # Remove duplicates by keeping the last Close price for each Date
        filtered_data = filtered_data.sort_values('Date').drop_duplicates('Date', keep='last')

        # Plot the data
        plt.plot(
            filtered_data['Date'],
            filtered_data['Close'],
            marker='o',
            linestyle='-',
            label=ticker
        )

    # Add labels, title, and legend
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Closing Price (USD)', fontsize=12, fontweight='bold')
    plt.title(f'Stock Comparison ({time_period})', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True, borderpad=1)
    plt.grid(False)
    plt.tight_layout()

    # Save the graph as a Base64-encoded image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300)  # High DPI for better quality
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return f"data:image/png;base64,{graph_url}"

@app.route('/api/stock_comparison', methods=['POST'])
def stock_comparison():
    """Compares stock data for specific tickers."""
    tickers = request.json.get('tickers', [])
    time_range = request.json.get('time_range', "7d")  # Default to 7 days if not provided

    if not tickers:
        return jsonify({'error': 'No tickers provided'}), 400

    # Validate time_range
    if time_range not in ["7d", "1m"]:
        return jsonify({"error": "Invalid time_range. Use '7d' or '1m'."}), 400

    # Load data for the specified tickers
    stock_data = {}
    df = load_stock_data(tickers)  # Pass tickers to load_stock_data()

    # Ensure the Date column is timezone-aware
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # Determine the start date based on the time range
    if time_range == "7d":
        start_date = datetime.now(pytz.utc) - timedelta(days=7)
    elif time_range == "1m":
        start_date = datetime.now(pytz.utc) - timedelta(days=30)

    # Filter data for each ticker
    for ticker in tickers:
        data = df[df["Ticker"] == ticker]
        if data.empty:
            return jsonify({'error': f'Stock {ticker} not found'}), 404

        # Filter data based on the time range
        filtered_data = data[data["Date"] >= start_date]
        if filtered_data.empty:
            return jsonify({'error': f'No data available for {ticker} in the specified time range.'}), 404

        stock_data[ticker] = filtered_data

    # Debug: Print filtered data for each ticker
    for ticker, data in stock_data.items():
        print(f"Filtered data for {ticker}:\n{data}")

    # Generate comparison chart
    comparison_chart = generate_comparison_chart(stock_data, time_period=time_range)
    
    # chart_file_path = generate_comparison_chart(stock_data, time_period=time_range)

    #  # Generate the URL for the chart
    # chart_url = request.host_url.rstrip("/") + "/" + chart_file_path.replace("\\", "/")
    return jsonify({
        'tickers': tickers,
        'comparison_chart': comparison_chart,
    })
    

@app.route("/api/market_sentiment", methods=["GET"])
def get_market_sentiment():
    """Returns market sentiment based on stock trends and volume."""
    time_range = request.args.get("time_range", "7d")  # Default to 7 days if not provided

    # Load stock data
    df = load_stock_data()
    print(df.head())  # Debug: Print the first few rows of the DataFrame

    # Filter data based on the time range
    if time_range == "7d":
        start_date = (datetime.now(pytz.utc) - timedelta(days=7)).replace(tzinfo=None)  # Remove timezone
    elif time_range == "1m":
        start_date = (datetime.now(pytz.utc) - timedelta(days=30)).replace(tzinfo=None)  # Remove timezone
    else:
        return jsonify({"error": "Invalid time_range. Use '7d' or '1m'."}), 400

    # Ensure the Date column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Convert Date column to datetime

    # Filter the DataFrame
    df = df[df["Date"] >= start_date]
    print(f"Data after filtering by time range ({time_range}):\n{df}")  # Debug: Print filtered data by time range

    # Check if the DataFrame is empty after filtering
    if df.empty:
        return jsonify({"error": f"No data available for the specified time range ({time_range})."}), 404

    # Calculate market sentiment
    df = calculate_trend(df)

    positive = sum(1 for _, row in df.iterrows() if "Upward" in row["Trend"])
    negative = sum(1 for _, row in df.iterrows() if "Downward" in row["Trend"])
    neutral = sum(1 for _, row in df.iterrows() if "Neutral" in row["Trend"])

    total = positive + negative + neutral
    market_sentiment = "Positive" if positive > negative else "Negative" if negative > positive else "Neutral"

    positive_pct = round((positive / total) * 100, 2)
    negative_pct = round((negative / total) * 100, 2)
    neutral_pct = round((neutral / total) * 100, 2)

    return jsonify({
        "positive": positive_pct,
        "negative": negative_pct,
        "neutral": neutral_pct,
        "overall_sentiment": market_sentiment
    })
    # Store in MySQL
    # conn = get_db_connection()
    # cursor = conn.cursor()
    # cursor.execute(
    #     "INSERT INTO market_sentiment (sentiment, positive_percentage, negative_percentage, neutral_percentage) VALUES (%s, %s, %s, %s)",
    #     (market_sentiment, positive_pct, negative_pct, neutral_pct)
    # )
    # conn.commit()
    # conn.close()

    # return jsonify({
    #     "positive": positive_pct,
    #     "negative": negative_pct,
    #     "neutral": neutral_pct,
    #     "overall_sentiment": market_sentiment
    # })
    
    
# --- API: Get All Tickers ---
@app.route("/api/tickers", methods=["GET"])
def get_all_tickers():
    """Returns a list of all unique stock tickers."""
    df = load_stock_data()
    tickers = df["Ticker"].unique().tolist()
    return jsonify({"tickers": tickers})

# --- Home Route ---
@app.route("/")
def home():
    return {"message": "Welcome to Stock Sentiment API"}

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(debug=True)