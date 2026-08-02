import os 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug
import streamlit as st

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
llm=ChatOpenAI(model='gpt-4o',api_key=api_key)

prompt_template=PromptTemplate(input_variables=['city','month','language','budget'],
                               template="""Welcome to the {city} travel guide!
                                        If you're visiting in {month}, here's what you can do:
                                        1. Must-visit attractions.
                                        2. Local cuisine you must try.
                                        3. Useful phrases in {language}.
                                        4. Tips for traveling on a {budget} budget.
                                        Enjoy your trip!""")

st.title("Travel_Guide")

city=st.text_input('Enter the city:')
month=st.date_input('Select a date:')
language=st.text_input('Enater the language:')
budget=st.selectbox('Travel budget',['low','medium','high'])

chain= prompt_template | llm
if city and month:
    response=chain.invoke( {"city":city,
                          "month":month,
                         "language":language,
                        "budget":budget})    
    st.write(response.content)