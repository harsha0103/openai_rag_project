import os
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()
key=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model='gpt-4o',api_key=key)

prompt_template=PromptTemplate(input_variables=["country",'number_of_paras','language'],
                               template="""You are an expert in traditional cuisines.
                               You provide information about a specific dish from a specific country.
                               Avoid giving information about fictional places. If the country is fictional
                               or non-existent answer: I don't know.Answer the question: 
                               
                               What is the traditional cuisine of {country}?
                               Answer in this many {number_of_paras} in {language}""")

st.title("Cuisine info : Prompt template")

user_input_country=st.text_input('Enter a country:')
user_input_paras=st.number_input('number of paragraphs:',min_value=1,max_value=5)
user_input_language=st.text_input("enter the language:")

if user_input_country:
    response=llm.invoke(prompt_template.format(country=user_input_country,
                                               number_of_paras=user_input_paras,
                                               language=user_input_language))
    st.write(response.content)

