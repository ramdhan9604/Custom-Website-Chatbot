import os
import torch
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain

# Set up environment variable for Groq API key
os.environ["GROQ_API_KEY"] = " "
os.environ["HUGGINGFACEHUB_API_TOKEN"] = " "


def load_documents(urls):
    """Load and return documents from the given URLs."""
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()

def split_text(documents, chunk_size=1000, chunk_overlap=100):
    """Split text into chunks for better processing."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_embeddings():
    """Initialize Hugging Face embeddings model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(text_chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = create_embeddings()
    return FAISS.from_documents(text_chunks, embeddings)

def query_llm(query, vector_store):
    """Retrieve the best answer from the vector store using a Groq-powered LLM."""

    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever = vector_store.as_retriever())
    result = qa_chain({"question": query},return_only_outputs=True)
    return result
