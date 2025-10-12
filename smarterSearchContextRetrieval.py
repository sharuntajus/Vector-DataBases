
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader


from langchain.retrievers.multi_query import MultiQueryRetriever
import logging


from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

import requests

#url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MZ9z1lm-Ui3YBp3SYWLTAQ/companypolicies.txt"
#response = requests.get(url)

#with open("companypolicies.txt", "wb") as f:
#   f.write(response.content)
pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf"
response = requests.get(pdf_url)

with open("langchain-paper.pdf", "wb") as f:
  f.write(response.content)

def llm():
    return OllamaLLM(model="mistral", temperature=0.5)

def local_embedding():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def text_splitter(data, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(data)


loader = TextLoader("companypolicies.txt")
txt_data = loader.load()

chunks_txt = text_splitter(txt_data, 200, 20)

vectordb = Chroma.from_documents(chunks_txt, embedding=local_embedding())

query = "email policy"

# Default retriever
retriever = vectordb.as_retriever()
docs = retriever.invoke(query)
print(docs)
# Top 1 result
retriever = vectordb.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke(query)
print(docs)

# MMR
retriever = vectordb.as_retriever(search_type="mmr")
docs = retriever.invoke(query)

# Score threshold
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4}
)
docs = retriever.invoke(query)
print(docs)


loader = PyPDFLoader("langchain-paper.pdf")
pdf_data = loader.load()

chunks_pdf = text_splitter(pdf_data, 500, 20)

# Replace old data in vectorstore
vectordb.delete(vectordb.get()["ids"])
vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=local_embedding())


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

query = "What does the paper say about langchain?"

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm()
)
docs = retriever.invoke(query)
print("Multi: ",docs)


docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
]

metadata_field_info = [
    AttributeInfo(name="genre", description="Genre of the movie", type="string"),
    AttributeInfo(name="year", description="Year of release", type="integer"),
    AttributeInfo(name="director", description="Director name", type="string"),
    AttributeInfo(name="rating", description="Rating 1-10", type="float"),
]

vectordb = Chroma.from_documents(docs, embedding=local_embedding())

retriever = SelfQueryRetriever.from_llm(
    llm=llm(),
    vectorstore=vectordb,
    document_contents="Brief summary of a movie.",
    metadata_field_info=metadata_field_info,
)

docs=retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
print("Self: ",docs)
