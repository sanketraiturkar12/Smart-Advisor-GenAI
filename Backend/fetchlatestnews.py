from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_openai import AzureChatOpenAI
import logging

def fetch_company_news_with_agent(company_name):
    """
    Fetch the latest news about a company using YahooFinanceNewsTool and LangChain agent.
    
    Args:
        company_name (str): The name of the company to search for.
    
    Returns:
        str: A summary of the latest news about the company.
    """
    logging.info(f"Fetching latest news for company: {company_name} using LangChain agent...")

    # Initialize the LLM and tools
    llm = AzureChatOpenAI(
        deployment_name="gpt-4",
        azure_endpoint="https://opneai-gpt-4o.openai.azure.com/",
        api_key="m9QfxXYxdW8W0KUVxXpBkmBvzgb0cPcLf6uQm81MKzaVw7lUIGl0JQQJ99AJACYeBjFXJ3w3AAABACOGwjQW",
        api_version="2024-12-01-preview",
        temperature=0
    )
    tools = [YahooFinanceNewsTool()]
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Use the agent to fetch news
    prompt = f"Get the latest news about {company_name}."
    result = agent_chain.invoke(prompt)

    logging.info(f"Fetched news for {company_name}: {result}")
    return result

if __name__ == "__main__":
    company_name = "Infosys"  # Example company name
    news_summary = fetch_company_news_with_agent(company_name)
    print(f"News Summary for {company_name}:\n{news_summary}")