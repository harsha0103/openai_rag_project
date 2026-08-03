import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate,ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


load_dotenv()

api_key=os.getenv('OPENAI_API_KEY')
embeddings=OpenAIEmbeddings(api_key=api_key)

document= TextLoader("product-data.txt").load()
text_splitter=RecursiveCharacterTextSplitter( chunk_size=200,chunk_overlap=10)
chunks=text_splitter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)
retriver=vector_store.as_retriever()


# Query embedding 

llm= ChatOpenAI(model='gpt-4o',api_key=api_key)
prompt_template=ChatPromptTemplate.from_messages ([
                        ('system',"""You are an assistant for answering questions.
                                Use the provided context to respond.If the answer
                                isn't clear, acknowledge that you don't know.
                                Limit your response to three concise sentences.
                                {context}"""),
                        
                        ('human',"""{input}""")
                    ])

qa_chain= create_stuff_documents_chain(llm,prompt_template)
rag_chain= create_retrieval_chain(retriver,qa_chain)

print("Chat with document")
Question=input('enter the input:')

if Question:
    response= rag_chain.invoke({'input':Question})
    print(response['answer'])