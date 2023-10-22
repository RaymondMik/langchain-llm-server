from flask import Blueprint, request
from dotenv import load_dotenv
import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.evaluation.qa import QAGenerateChain

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')

evaluation_route = Blueprint('evaluation_route', __name__)

@evaluation_route.route('/evaluation')
def evaluation():
    file = './assets/todos.csv'
    loader = CSVLoader(file_path=file)
    data = loader.load()

    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])

    llm = ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo', openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=index.vectorstore.as_retriever(), 
        verbose=True,
        chain_type_kwargs = {
            "document_separator": "<<<<>>>>>"
        }
    )

    print(data[10])

    example_gen_chain = QAGenerateChain.from_llm(llm)

    new_examples = example_gen_chain.apply_and_parse(
        [{"doc": t} for t in data[:5]]
    )

    new_examples[0]

    