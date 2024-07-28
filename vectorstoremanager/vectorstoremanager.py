from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import os
from utils import clean_text
class VectorStoreManager:
    """
    VectorStoreManager handles loading, splitting, and managing vector stores for document embeddings.

    Attributes:
        pdf_directory (str): Directory containing PDF files to be processed.
        model_name (str): The name of the model to be used for embeddings.
        save_path (str): Path to save the vector store.
        vector_store (FAISS): The vector store instance.
        text_splitter (RecursiveCharacterTextSplitter): An instance of RecursiveCharacterTextSplitter for splitting text.

    Methods:
        load_and_split_pdfs(): Loads and splits PDF files into text chunks.
        create_vector_store(chunks: list): Creates a vector store from text chunks.
        load_vector_store(): Loads an existing vector store or creates a new one if it doesn't exist.
        get_vector_store(): Retrieves the vector store.
    """
    def __init__(self, pdf_directory=None, model_name="llama3.1:latest", save_path=None, text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)):
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        self.save_path = save_path
        self.vector_store = None
        self.text_splitter = text_splitter

    def load_and_split_pdfs(self):
        chunks = []
        pdf_files = glob.glob(f"{self.pdf_directory}/*.pdf")
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            for page in pages:
                for chunk in self.text_splitter.split_text(page.page_content):
                    chunk = clean_text(chunk)
                    chunks.append({"text": chunk, "metadata": page.metadata})
        return chunks

    def create_vector_store(self, chunks):
        embedding = OllamaEmbeddings(model=self.model_name)
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        self.vector_store = FAISS.from_texts(texts, embedding=embedding, metadatas=metadatas)
        if self.save_path:
            self.vector_store.save_local(self.save_path)

    def load_vector_store(self):
        embedding = OllamaEmbeddings(model=self.model_name)
        if self.save_path and os.path.exists(self.save_path):
            self.vector_store = FAISS.load_local(self.save_path, embeddings=embedding, allow_dangerous_deserialization=True)
        else:
            if not self.pdf_directory:
                raise ValueError("PDF directory must be provided to create a new vector store.")
            chunks = self.load_and_split_pdfs()
            self.create_vector_store(chunks)

    def get_vector_store(self):
        if not self.vector_store:
            self.load_vector_store()
        return self.vector_store
