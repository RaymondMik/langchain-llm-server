from flask import Blueprint
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm_chain_route = Blueprint('llm_chain_route', __name__)

@llm_chain_route.route("/llm_chain")
def llm_chain():
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe \
        a company that makes {product}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    product = "Queen Size Sheet Set"
    
    return chain.run(product)