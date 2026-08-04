import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_chroma import Chroma

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever

from langchain_openai import ChatOpenAI,OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
embedding=OpenAIEmbeddings(api_key=api_key)

@st.cache_resource
def build_rag_chain():
    #load the document, split and embed
    document=TextLoader('resources/legal_data /Legal_Document_Analysis_Data.txt').load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
    chunks=text_splitter.split_documents(document)
    vector_store=Chroma.from_documents(chunks,embedding)
    retriver=vector_store.as_retriever()
    return retriver

retriver=build_rag_chain()


llm=ChatOpenAI(model='gpt-4o',api_key=api_key)
prompt_template=ChatPromptTemplate.from_messages ([('system',"""You are an assistant for answering questions.
                                                    Use the provided context to respond.If the answer
                                                    isn't clear, acknowledge that you don't know.
                                                    Limit your response to three concise sentences.
                                                    {context}"""),
                        MessagesPlaceholder(variable_name='chat_history'),
                        ('human',"""{input}""")
                    ])

contextualize_q_prompt = ChatPromptTemplate.from_messages([("system","""You are a helpful assistant that reformulates follow-up
                                                            questions into standalone questions.Use the chat history and the latest user input to create
                                                            a self-contained question.Do NOT answer the question, only rewrite it."""),
                                                        MessagesPlaceholder("chat_history"),
                                                        ("human", "{input}"),
                                                    ]
                                                )

historical_aware_retreat= create_history_aware_retriever(llm,retriver,contextualize_q_prompt)
qa_chain=create_stuff_documents_chain(llm,prompt_template)
rag_chain=create_retrieval_chain(historical_aware_retreat,qa_chain)


st.title("Legal Document")
question=st.text_input("enter your question:")

history_of_messages=StreamlitChatMessageHistory()
chain_with_history=RunnableWithMessageHistory(rag_chain,
                           lambda session_id:history_of_messages,
                           input_messages_key='input',
                           history_messages_key='chat_history',
                           output_messages_key='answer')

if question:
    response=chain_with_history.invoke({'input':question},{'configurable':{'session_id':'123'}})
    st.write(response['answer'])

st.write('history')
st.write(history_of_messages)


