from flask import Blueprint, request
from dotenv import load_dotenv
import os

import langchain
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')

agent_route = Blueprint('agent_route', __name__)

@agent_route.route("/agent")
def agent():  
    query = request.args.get("query")
    print("QUERY", query)
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    # tools = load_tools(["llm-math","wikipedia"], llm=llm)
    tools = load_tools(["llm-math","serpapi"], llm=llm)

    agent= initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True)
    
    return agent(query)

python_agent_route = Blueprint('python_agent_route', __name__)

@python_agent_route.route("/python_agent")
def python_agent():  
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    agent = create_python_agent(
        llm,
        tool=PythonREPLTool(),
        verbose=True,
        handle_parsing_errors=True
    )

    customer_list = [
                ["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
    
    langchain.debug=True
    result = agent.run(f"""Sort these customers by \
        last name and then first name \
        and print the output: {customer_list}""") 
    langchain.debug=False

    return result

agent_route = Blueprint('agent_route', __name__)

@agent_route.route("/agent")
def agent():  
    query = request.args.get("query")
    print("QUERY", query)
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    # tools = load_tools(["llm-math","wikipedia"], llm=llm)
    tools = load_tools(["llm-math","serpapi"], llm=llm)

    agent= initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True)
    
    return agent(query)

# tool decorator, it can be applied to any function to transform it in a tool
from langchain.agents import tool
from langchain.utilities import TextRequestsWrapper
from datetime import date

custom_agent_route = Blueprint('custom_agent_route', __name__)

@custom_agent_route.route("/custom_agent")
def custom_agent():  
    query = request.args.get("query")
    requests = TextRequestsWrapper()

    # @tool
    # def time(text: str) -> str:
    #     f"""Returns todays date, use this for any \
    #     questions related to knowing todays date. \
    #     The input should always be an empty string, \
    #     and this function will always return todays \
    #     date - any date mathmatics should occur \
    #     outside this function.{query}"""
    #     return str(date.today())
    
    @tool
    def api_request(text: str) -> str:
        """Return the value provided by the below GET HTTP request. \
           The value returned the request is always a Python dictionary."""
        return requests.get("https://webhook.site/54511a40-b58a-49f9-81cd-94cb5780d78a")

    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    tools = load_tools(["llm-math"], llm=llm)

    agent= initialize_agent(
        tools + [api_request], 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True)
    
    langchain.debug=True
    
    try:
        result = agent(query) 
    except: 
        print("exception on external access")
  
    langchain.debug=False

    return result