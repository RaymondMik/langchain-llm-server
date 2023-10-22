from flask import Blueprint, request
from dotenv import load_dotenv
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
retrieval_route = Blueprint('retrieval_route', __name__)

@retrieval_route.route('/retrieval')
def retrieval():
    query = request.args.get('query')
   
    loader = TextLoader('./assets/bike_description.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    raw_documents = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(raw_documents, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo', openai_api_key=openai_api_key)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)
