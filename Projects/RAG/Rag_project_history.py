import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate,ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory 

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever


load_dotenv()

api_key=os.getenv('OPENAI_API_KEY')
embeddings=OpenAIEmbeddings(api_key=api_key)


@st.cache_resource
def build_rag_chain():
    document = TextLoader("product-data.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriver = vector_store.as_retriever()
    return retriver

retriver = build_rag_chain()

# Query embedding 

llm= ChatOpenAI(model='gpt-4o',api_key=api_key)

prompt_template=ChatPromptTemplate.from_messages ([
                        ('system',"""You are an assistant for answering questions.
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


history_aware_retrive = create_history_aware_retriever(llm, retriver, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(llm, prompt_template)  # this one keeps {context} — correct
rag_chain = create_retrieval_chain(history_aware_retrive, qa_chain)



st.title("Chat with document")
question=st.text_input('enter the input:')

history_of_messages=StreamlitChatMessageHistory()

chain_with_history=RunnableWithMessageHistory(
                        rag_chain,
                        lambda session_id: history_of_messages,
                        input_messages_key='input',
                        history_messages_key='chat_history',    
                        output_messages_key='answer',   # <-- add this
                    )

if question:
    response= chain_with_history.invoke({'input':question},{'configurable':{'session_id':'123'}})
    st.write(response['answer'])

st.write('history')
st.write(history_of_messages)
