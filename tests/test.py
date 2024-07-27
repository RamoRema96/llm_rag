from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = PyPDFLoader("data/Omar_Amer_cv.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=20)
chunks = []

for i, chunk in enumerate(text_splitter.split_text(pages[0].page_content)):
    chunks.append(

        {"text":chunk,
         "metadata":pages[0].metadata}
    )
print(chunks)
# print(dir(pages[0]))
# print(pages[0].metadata)
# print(pages[0].page_content)