from flask import Blueprint
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

sequential_chain_route = Blueprint('sequential_chain_route', __name__)

# single I/O chain
@sequential_chain_route.route("/sequential_chain")
def sequential_chain():
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    # prompt template 1: translate to english
    first_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to english:"
        "\n\n{Review}"
    )
    # chain 1: input= Review and output= English_Review
    chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                        output_key="English_Review"
                        )
    
    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:"
        "\n\n{English_Review}"
    )
    # chain 2: input= English_Review and output= summary
    chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                        output_key="summary"
                        )

    # prompt template 3: translate to english
    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    # chain 3: input= Review and output= language
    chain_three = LLMChain(llm=llm, prompt=third_prompt,
                        output_key="language"
                        )

    # prompt template 4: follow up message
    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following "
        "summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    # chain 4: input= summary, language and output= followup_message
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                        output_key="followup_message"
                        )

    # overall_chain: input= Review 
    # and output= English_Review,summary, followup_message
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three, chain_four],
        input_variables=["Review"],
        output_variables=["English_Review", "summary","followup_message"],
        verbose=True)
    
    review = "Ottimo prodotto. Impermeabile, comodo, davvero un fantastico paio di scarpe"
    
    return overall_chain(review)