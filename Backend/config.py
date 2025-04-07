# Configuration file for Azure OpenAI & MySQL
 
AZURE_OPENAI_ENDPOINT = "https://opneai-gpt-4o.openai.azure.com/"
AZURE_OPENAI_KEY = "m9QfxXYxdW8W0KUVxXpBkmBvzgb0cPcLf6uQm81MKzaVw7lUIGl0JQQJ99AJACYeBjFXJ3w3AAABACOGwjQW"
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"  # Model deployment name in Azure OpenAI
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
 
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "stock_sentiment_db",
}