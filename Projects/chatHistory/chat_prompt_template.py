import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
import streamlit as st

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
llm=ChatOpenAI(model='gpt-4o',api_key=api_key)

chatprompt=ChatPromptTemplate.from_messages( [
                                            ("system",'you are a ajile coach. answer any questions related to ajile process'),
                                            ("human",'{input}')
                                        ])

st.title("Agile guide")
input=st.text_input("input:")

chain= chatprompt | llm | StrOutputParser()

if input:
    response= chain.invoke({'input':input})
    st.write(response)