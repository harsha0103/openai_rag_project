import os 
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st 

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model='gpt-4o',api_key=api_key)

title_prompt=PromptTemplate(
                input_variables=['topic'],
                template="""You are an experienced speech writer.
                        You need to craft an impactful title for a speech
                        on the following topic: {topic}
                        Answer exactly with one title. """)

speach_prompt= PromptTemplate(
                input_variables=['title','emotion'],
                template="""You need to write a powerful {emotion} speech of 100 words
                            for the following title: {title}
                            format the output with 2 keys: 'title','speach',fill them with thier values """)


chain1 = title_prompt | llm | StrOutputParser() | (lambda title:(st.write(title),title)[1])
chain2 = speach_prompt | llm 
final_chain= chain1 | (lambda title: {'title':title,'emotion':emotion}) | chain2 | JsonOutputParser()

st.title(" Simple chain")

topic=st.text_input('Enter a topic:')
emotion=st.text_input('Enter a emotion:')


if topic:
    response=final_chain.invoke({"topic":topic})

    st.write(response)
