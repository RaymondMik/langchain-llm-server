from flask import Blueprint
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

simple_sequential_chain_route = Blueprint('simple_sequential_chain_route', __name__)

# single I/O chain
@simple_sequential_chain_route.route('/simple_sequential_chain')
def simple_sequential_chain():
    llm = ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo', openai_api_key=openai_api_key)
    product = 'Queen Size Sheet Set'

    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        'What is the best name to describe \
        a company that makes {product}?'
    )

    # Chain 1
    chain_one = LLMChain(llm=llm, prompt=first_prompt)

    # prompt template 2
    second_prompt = ChatPromptTemplate.from_template(
        'Write a 20 words description for the following \
        company:{company_name}'
    )
    # chain 2
    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
      
    return overall_simple_chain.run(product)