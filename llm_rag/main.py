
import sys
import os

# Add the root directory of your project to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from vectorialdb.vectorialdb import VectorStore
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils import chunk_cv
import pandas as pd 

# Read the data
cvs_df = pd.read_csv("./data/synthetic_cv.csv")

# chunked the data
chunked_cvs = cvs_df.apply(chunk_cv, axis=1).explode().reset_index(drop=True)

# Convert to a list for the embedding
docs = [{"text": chunk["text"], "metadata": chunk["metadata"]} for chunk in chunked_cvs]


private_model = "llama3.1:latest"
# Initialize the class
vectordb = VectorStore(embedding_model = private_model, save_path="./faiss_index_cv")

# Load or create the db
vectorStore = vectordb.load_or_create_db(docs=docs)

def create_chain(vectorStore):

    # Instatiate the model
    llm = Ollama(
        model=private_model,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.4,
    )
    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
    You are an expert in talent acquisition that helps determine the best
    candidate among multiple suitable resumes.
    Use only the following pieces of context to determine the best resume
    given a job description.
    You should provide some detailed explanations for the best resume choice.
    Make sure to also return a detailed summary of the original text of
    the best resume.
    Because there can be applicants with similar names, try to use from the metadat the id
    to refer to resumes in your response, otherwise use the name.
    If you don't know the answer, just say that you don't know, do not try to
    make up an answer.
    Context: {context}
    Question: {input}
    """
    )
    # Create LLM chain
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    # se vuoi che ti funzioni devi per forza mettere quella variabile context
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})
    retriever_chain = create_retrieval_chain(retriever, chain)
    return retriever_chain

chain = create_chain(vectorStore)
response = chain.invoke(
    {
        "input": "Job description: We are searching for a data scientist with at least 6 years of experience. It must be profiency in python and machine learning"
    }
)
print(response)