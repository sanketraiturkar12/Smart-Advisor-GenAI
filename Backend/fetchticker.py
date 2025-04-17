import requests
import logging

def get_ticker(company_name):
    """
    Fetch the ticker symbol for a company using Yahoo Finance API.

    Args:
        company_name (str): The name of the company to search for.

    Returns:
        str: The ticker symbol of the company, or None if not found.
    """
    logging.info(f"Fetching ticker for company: {company_name}...")
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    try:
        res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
        res.raise_for_status()
        data = res.json()

        if "quotes" in data and len(data["quotes"]) > 0:
            company_code = data['quotes'][0]['symbol']
            logging.info(f"Ticker found for {company_name}: {company_code}")
            return company_code
        else:
            logging.warning(f"No ticker found for {company_name}.")
    except Exception as e:
        logging.error(f"Error fetching ticker for {company_name}: {e}")
    return None

if __name__ == "__main__":
    company_name = "Infosys"  # Example company name
    ticker = get_ticker(company_name)
    if ticker:
        print(f"The ticker for {company_name} is {ticker}.")
    else:
        print(f"Could not find a ticker for {company_name}.")