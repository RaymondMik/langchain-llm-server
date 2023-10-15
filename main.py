from flask import Flask
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

from langchain.chat_models import ChatOpenAI

# RETRIEVAL
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

# ROUTES
from routes.llm_chain import llm_chain_route
from routes.memories import memories_route
from routes.simple_sequential_chain import simple_sequential_chain_route
from routes.sequential_chain import sequential_chain_route
from routes.router_chain import router_chain_route
from routes.agent import agent_route
from routes.agent import python_agent_route
from routes.agent import custom_agent_route

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
app.register_blueprint(llm_chain_route)
app.register_blueprint(memories_route)
app.register_blueprint(simple_sequential_chain_route)
app.register_blueprint(sequential_chain_route)
app.register_blueprint(router_chain_route)
app.register_blueprint(agent_route)
app.register_blueprint(python_agent_route)
app.register_blueprint(custom_agent_route)

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/")
def hello_world():
    return 'hello world'

@app.route("/retrieval")
def retrieval():
    file = './assets/todos.csv'
    loader = CSVLoader(file_path=file)  
    docs = loader.load()
    docs[0]
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
    )
    query = "Please tell me what's the first item on my list?"
    docs = db.similarity_search(query)
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
    )
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embeddings,
    ).from_loaders([loader])
    return qa_stuff.run(query)

