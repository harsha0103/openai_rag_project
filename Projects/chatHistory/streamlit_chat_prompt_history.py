import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
#runnable message history with streamlit
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

import streamlit as st

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
llm=ChatOpenAI(model='gpt-4o',api_key=api_key)

chatprompt=ChatPromptTemplate.from_messages( [
                                            ("system",'you are a ajile coach. answer any questions related to ajile process'),
                                            MessagesPlaceholder(variable_name='chat_history'),
                                            ("human",'{input}')
                                        ])

st.title("Agile guide with history")
input=st.text_input("input:")

chain= chatprompt | llm | StrOutputParser()

history_of_messages=StreamlitChatMessageHistory()

chain_with_history=RunnableWithMessageHistory(
                        chain,
                        lambda session_id: history_of_messages,
                        input_messages_key='input',
                        history_messages_key='chat_history'
                    )


if input:
    response= chain_with_history.invoke( {'input':input},{'configurable':{'session_id': 'abc_123'}})
    st.write(response)

st.write("HISTORY")
st.write(history_of_messages)