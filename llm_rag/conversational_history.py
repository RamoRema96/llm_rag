import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

# Add the root directory of your project to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from chainmanager.chainmanager import ChainManager
from vectorstoremanager.vectorstoremanager import VectorStoreManager

pdf_directory="book"
save_path = "itps.faiss"
model_name = "llama3.1:latest"

# Create and/or load the vector store
vector_store_manager = VectorStoreManager(pdf_directory=pdf_directory, save_path=save_path, model_name=model_name)
vector_store_manager.load_vector_store()
print("Vector store loaded or created")
# Create the chain manager and process a query
chain_manager = ChainManager(vector_store_manager, model_name=model_name)
chat_history = []


while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    response = chain_manager.process_chat(query=user_input, chat_history=chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    print("Assistant:", response)