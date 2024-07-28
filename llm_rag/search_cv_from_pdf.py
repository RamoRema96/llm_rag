import sys
import os

# Add the root directory of your project to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from chainmanager.chainmanager import ChainManager
from vectorstoremanager.vectorstoremanager import VectorStoreManager

pdf_directory = "data"
save_path = "vector_store_cvs.faiss"

# Create and/or load the vector store
vector_store_manager = VectorStoreManager(
    pdf_directory=pdf_directory, save_path=save_path
)
vector_store_manager.load_vector_store()

# Create the chain manager and process a query
chain_manager = ChainManager(vector_store_manager)
response = chain_manager.process_query(
    "I need a data scientist, Which is the most suitable candidate? what do you think of Omar Amer"
)
print(response)
