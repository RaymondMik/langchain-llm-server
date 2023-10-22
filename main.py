from flask import Flask
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

# ROUTES
from routes.llm_chain import llm_chain_route
from routes.memories import memories_route
from routes.simple_sequential_chain import simple_sequential_chain_route
from routes.sequential_chain import sequential_chain_route
from routes.router_chain import router_chain_route
from routes.retrieval import retrieval_route
from routes.agent import agent_route
from routes.agent import python_agent_route
from routes.agent import custom_agent_route
from routes.evaluation import evaluation_route

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
app.register_blueprint(llm_chain_route)
app.register_blueprint(memories_route)
app.register_blueprint(simple_sequential_chain_route)
app.register_blueprint(sequential_chain_route)
app.register_blueprint(router_chain_route)
app.register_blueprint(retrieval_route)
app.register_blueprint(agent_route)
app.register_blueprint(python_agent_route)
app.register_blueprint(custom_agent_route)
app.register_blueprint(evaluation_route)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def hello_world():
    return 'hello world'
