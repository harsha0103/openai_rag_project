import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv()
key=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o",api_key=key)

st.title("Ask anything")

question=st.text_input("Ask your question:")

if question:
    response=llm.invoke(question)
    st.write(response.content)