from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class ChainManager:
    def __init__(self, vector_store_manager, model_name="llama3.1:latest"):
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name

    def create_chain(self):
        llm = Ollama(
            model=self.model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
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