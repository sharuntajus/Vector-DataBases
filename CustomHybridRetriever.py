import warnings
import numpy as np
from typing import List, Optional
import llama_index
#from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
warnings.filterwarnings("ignore")

import chromadb
from chromadb.utils import embedding_functions

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser

# HuggingFace embeddings
#from langchain_huggingface import HuggingFaceEmbeddings
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain.embeddings import HuggingFaceEmbeddings
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
def get_llm():
    #return Ollama(model="mistral", temperature=0.5)
    return OllamaLLM(model="mistral", temperature=0.5)


Settings.llm=get_llm()
# Create LangChain HF embeddings

#embed_model = LangchainEmbedding(hf_embed)
Settings.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#Settings.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning",
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}

class AdvancedRetrieversLab:
    def __init__(self):
        self.documents = [Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes=SentenceSplitter().get_nodes_from_documents(self.documents)

        self.vector_index=VectorStoreIndex.from_documents(self.documents)
        self.document_summary_index=DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index=KeywordTableIndex.from_documents(self.documents)


lab = AdvancedRetrieversLab()
vector_retriever = lab.vector_index.as_retriever(similarity_top_k=10)
keyword_retriever = lab.keyword_index.as_retriever(similarity_top_k=10)
#
# try:
#     bm25_retriever = BM25Retriever.from_defaults(
#         nodes=lab.nodes, similarity_top_k=10
#     )
# except:
#     # Fallback if BM25 is not available
#     bm25_retriever = vector_retriever


def hybrid_retrieve(query, top_k=5):
    # Get results from both retrievers
    vector_results = vector_retriever.retrieve(query)
    #bm25_results = bm25_retriever.retrieve(query)
    keyword_results = keyword_retriever.retrieve(query)
    # Create dictionaries using text content as keys (since node IDs differ)
    vector_scores = {}
    bm25_scores = {}
    keyword_scores={}
    all_nodes = {}

    # Normalize vector scores
    max_vector_score = max([r.score for r in vector_results]) if vector_results else 1
    for result in vector_results:
        text_key = result.text.strip()  # Use text content as key
        normalized_score = result.score / max_vector_score
        vector_scores[text_key] = normalized_score
        all_nodes[text_key] = result

    # Normalize BM25 scores
    # max_bm25_score = max([r.score for r in bm25_results]) if bm25_results else 1
    # for result in bm25_results:
    #     text_key = result.text.strip()  # Use text content as key
    #     normalized_score = result.score / max_bm25_score
    #     bm25_scores[text_key] = normalized_score
    #     all_nodes[text_key] = result
    max_keyword = max([r.score for r in keyword_results]) if keyword_results else 1
    for r in keyword_results:
        key = r.text.strip()
        keyword_scores[key] = r.score / max_keyword
        all_nodes[key] = r

    # Calculate hybrid scores
    hybrid_results = []
    for text_key in all_nodes:
        vector_score = vector_scores.get(text_key, 0)
        k = keyword_scores.get(text_key, 0)
        #bm25_score = bm25_scores.get(text_key, 0)
        hybrid_score = 0.7 * vector_score + 0.3 * k #bm25_score

        hybrid_results.append({
            'node': all_nodes[text_key],
            'vector_score': vector_score,
            #'bm25_score': bm25_score,
            "keyword_score": k,
            'hybrid_score': hybrid_score
        })

    # Sort by hybrid score and return top k
    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_results[:top_k]


# Test with different queries
test_queries = [
    "What is machine learning?",
    "neural networks deep learning",
    "supervised learning techniques"
]

for query in test_queries:
    print(f"Query: {query}")
    results = hybrid_retrieve(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"   Vector: {result['vector_score']:.3f}, KW: {result['keyword_score']:.3f} ");#BM25: {result['bm25_score']:.3f}")
        print(f"   Text: {result['node'].text[:80]}...")
    print()