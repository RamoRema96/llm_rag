from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from vectorstoremanager.vectorstoremanager import VectorStoreManager


class ChainManager:
    """
    ChainManager is responsible for creating and managing chains for processing queries
    using a language model.

    Attributes:
        vector_store_manager (VectorStoreManager): An instance of VectorStoreManager to handle vector store operations.
        model_name (str): The name of the model to be used.

    Methods:
        create_chain(): Creates a retrieval chain for selecting the best resume based on a job description.
        process_query(query: str): Processes a query to determine the best resume.
        create_chain_conversational(): Creates a conversational chain for answering physics-related questions.
        process_chat(query: str, chat_history: list): Processes a conversational query with chat history.
    """

    def __init__(self, vector_store_manager:VectorStoreManager, model_name="llama3.1:latest"):
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name

    def create_chain(self):
        #TODO The prompt should be passed outside
        llm = Ollama(
            model=self.model_name,
            # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.4,
        )
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in talent acquisition that helps determine the best
            candidate among multiple suitable resumes.
            Use only the following pieces of context to determine the best resume
            given a job description.
            You should provide some detailed explanations for the best resume choice.
            Make sure to also return a detailed summary of the original text of
            the best resume.
            Because there can be applicants with similar names, try to use from the metadata the id
            to refer to resumes in your response, otherwise use the name.
            If you don't know the answer, just say that you don't know, do not try to
            make up an answer.
            In any case, answer precisely to the user question, if he does not ask about candidate explicitely, you don't talk about it
            Context: {context}
            Question: {input}
            """
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        vector_store = self.vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retriever_chain = create_retrieval_chain(retriever, chain)
        return retriever_chain

    def process_query(self, query):
        chain = self.create_chain()
        response = chain.invoke({"input": query})
        return response

    def create_chain_conversational(self):
        #TODO the propmpt should be passed outside
        llm = Ollama(
            model=self.model_name,
            # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.4,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are one of the best Physics professor of University at
            La Sapienza. You are teaching the course base on the book at the university. Answer to the students questions, they are also physicist at the 4 year: {context}
            """,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        vector_store = self.vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                (
                    "human",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm=llm, retriever=retriever, prompt=retriever_prompt
        )
        retriever_chain = create_retrieval_chain(history_aware_retriever, chain)
        return retriever_chain

    def process_chat(self, query, chat_history):
        chain = self.create_chain_conversational()
        response = chain.invoke({"input": query, "chat_history": chat_history})
        return response["answer"]
