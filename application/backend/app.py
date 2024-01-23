from flask import Flask, request
import os
import openai
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.indexes import SQLRecordManager, index
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain.schema.messages import get_buffer_string
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema import format_document
from operator import itemgetter


_ = load_dotenv(find_dotenv())
openai.api_key  = os.environ['OPENAI_API_KEY']
host = os.getenv("PG_VECTOR_HOST")
user = os.getenv("PG_VECTOR_USER")
password = os.getenv("PG_VECTOR_PASSWORD")
database = os.getenv("PGDATABASE")
COLLECTION_NAME = "langchain_collection"
CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}"

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = PGVector(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

namespace = f"pgvector/{COLLECTION_NAME}"
record_manager = SQLRecordManager(
    namespace, db_url=CONNECTION_STRING
)

retriever = vector_store.as_retriever()

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


template = """Answer the question based only on the following context:
{context}
Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()


app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test_endpoint():
    return 'works'

@app.route('/fileEmbed', methods=['GET'])
def load_file():
    loader = PyPDFLoader("docs/MLInterview.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
    chunks = text_splitter.split_documents(pages)

    record_manager.create_schema()
    index(
        chunks,
        record_manager,
        vector_store,
        cleanup=None,
        source_id_key="source",
    )
    return "embeddings created"

@app.route('/askQuestion', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    # retriever.get_relevant_documents(question, k=5)
    answer = conversational_qa_chain.invoke(
    {
        "question": question,
        "chat_history": [],
    }
    )
    return answer


if __name__ == '__main__':
    app.run(debug=True)