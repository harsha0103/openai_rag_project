import os 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug
import streamlit as st

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
llm=ChatOpenAI(model='gpt-4o',api_key=api_key)

prompt_template=PromptTemplate(input_variables=['position','company','strengths','weaknesses'],
                               template="""You are a career coach. Provide tailored interview tips for the
                                        position of {position} at {company}.
                                        Highlight your strengths in {strengths} and prepare for questions
                                        about your weaknesses such as {weaknesses}.""")

st.title("Travel_Guide")

position=st.text_input('Enter the position:')
company=st.text_input('Enter a company:')
strengths=st.text_area('Enater your strengths:',height=100)
weaknesses=st.text_area('Enter your weakness:',height=100)

if position and company:
    response=llm.invoke(prompt_template.format(position=position,
                                               company=company,
                                               strengths=strengths,
                                               weaknesses=weaknesses))
    st.write(response.content)