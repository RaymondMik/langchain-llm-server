from flask import Blueprint
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

memories_route = Blueprint('memories_route', __name__)

@memories_route.route('/memories')
def memories():
    schedule = 'There is a meeting at 8am with your product team. \
        You will need your powerpoint presentation prepared. \
        9am-12pm have time to work on your LangChain \
        project which will go quickly because Langchain is such a powerful tool. \
        At Noon, lunch at the italian resturant with a customer who is driving \
        from over an hour away to meet you to understand the latest in AI. \
        Be sure to bring your laptop to show the latest LLM demo.'
    # memory = ConversationBufferMemory()
    # memory = ConversationBufferWindowMemory(k=1) 
    # memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
    llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo', openai_api_key=openai_api_key)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    memory.save_context({'input': 'Hello'}, {'output': 'Whats up'})
    memory.save_context({'input': 'Not much, just hanging'},
                        {'output': 'Cool'})
    memory.save_context({'input': 'What is on the schedule today?'}, 
                        {'output': f'{schedule}'})

    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=True
    )

    conversation.predict(input='Hi, my name is Andrew')
    conversation.predict(input='What is 1+1?')
    conversation.predict(input='When do I have time to work on LangChain?')

    return memory.load_memory_variables({})