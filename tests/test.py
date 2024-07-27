from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import glob


text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
chunks = []

pdf_files = glob.glob("data/*.pdf")
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    # Iterate through all pages and split each page's content into chunks
    for page in pages:
        for chunk in text_splitter.split_text(page.page_content):
            chunks.append({"text": chunk, "metadata": page.metadata})


private_model = "llama3.1:latest"
embedding = OllamaEmbeddings(model=private_model)
texts = [chunk["text"] for chunk in chunks]
metadatas = [chunk["metadata"] for chunk in chunks]

vectorStore = FAISS.from_texts(
                texts, embedding=embedding, metadatas=metadatas
            )

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
        "input": "I need a data scientist, Which is the most suitable candidate?"
    }
)
print(response)
# print(chunks)
# print(dir(pages[0]))
# print(pages[0].metadata)
# print(pages[0].page_content)