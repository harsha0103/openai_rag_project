import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import numpy as np

load_dotenv()

api_key=os.getenv('OPENAI_API_KEY')
llm=OpenAIEmbeddings(api_key=api_key)

document= TextLoader("job_listings.txt").load()
text_splitter=RecursiveCharacterTextSplitter( chunk_size=200,chunk_overlap=10)
chunks=text_splitter.split_documents(document)
db=Chroma.from_documents(chunks,llm)
retriver=db.as_retriever()

text=input('enter the input:')
# embedding_vector=llm.embed_query(text)
# docs=db.similarity_search_by_vector(embedding_vector)
docs=retriver.invoke(text)


for doc in docs:
    print(doc.page_content)