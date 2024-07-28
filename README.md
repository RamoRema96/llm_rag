
# Chatbot and RAG with private model

This project utilizes language models to build a chatbot with access to the pdf files loaded. We can use also just to retrieve the most relavant document instead of having a true conversation

## Directory Structure

```
.
├── Dockerfile
├── README.md
├── __pycache__
│   └── utils.cpython-311.pyc
├── book
│   └── itps.pdf
├── chainmanager
│   ├── __pycache__
│   │   └── chainmanager.cpython-311.pyc
│   └── chainmanager.py
├── data
│   ├── Alice_Johnson_CV.pdf
│   ├── Bob_Smith_CV.pdf
│   ├── Charlie_Brown_CV.pdf
│   ├── Dana_White_CV.pdf
│   ├── Omar_Amer_cv.pdf
│   └── synthetic_cv.csv
├── itps.faiss
│   ├── index.faiss
│   └── index.pkl
├── llm_rag
│   ├── __init__.py
│   ├── conversational_history.py
│   ├── main.py
│   └── search_cv_from_pdf.py
├── poetry.lock
├── pyproject.toml
├── run_ollama.sh
├── start_ollama.sh
├── tests
│   ├── __init__.py
│   └── test.py
├── utils.py
├── vector_store_cvs.faiss
│   ├── index.faiss
│   └── index.pkl
├── vectorialdb
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   └── vectorialdb.cpython-311.pyc
│   └── vectorialdb.py
├── vectorstoremanager
│   ├── __pycache__
│   │   └── vectorstoremanager.cpython-311.pyc
│   └── vectorstoremanager.py
└── wait_for_ollama.sh
```

## Setup

### Prerequisites

- Python 3.11
- [Poetry](https://python-poetry.org/) for managing dependencies

### Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

### Running the Project

1. Navigate to the \`llm_rag\` directory and run the main script:

   ```bash
   cd llm_rag
   poetry run python main.py
   ```

2. Interact with the system via the command line. To exit the interactive loop, type \`exit\`.

## Usage

### VectorStoreManager

The \`VectorStoreManager\` class handles loading, splitting, and managing vector stores for document embeddings.

#### Methods:

- \`load_and_split_pdfs()\`: Loads and splits PDF files into text chunks.
- \`create_vector_store(chunks)\`: Creates a vector store from text chunks.
- \`load_vector_store()\`: Loads an existing vector store or creates a new one if it doesn't exist.
- \`get_vector_store()\`: Retrieves the vector store.

### ChainManager

The \`ChainManager\` class is responsible for creating and managing chains for processing queries using a language model.

#### Methods:

- \`create_chain()\`: Creates a retrieval chain for selecting the best resume based on a job description.
- \`process_query(query)\`: Processes a query to determine the best resume.
- \`create_chain_conversational()\`: Creates a conversational chain for answering physics-related questions.
- \`process_chat(query, chat_history)\`: Processes a conversational query with chat history.



## Docker

To build and run the project using Docker, follow these steps:

1. Build the Docker image:

   ```bash
   docker build -t project-image .
   ```

2. Run the Docker container:

   ```bash
   docker run -it project-image
   ```

## Testing

Tests are located in the \`tests\` directory. To run the tests, use the following command:

```bash
poetry run pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Poetry](https://python-poetry.org/)
