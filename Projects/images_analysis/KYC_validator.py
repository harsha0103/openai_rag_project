from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import base64
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

def encode_image(upload_file):
    return base64.b64encode(upload_file.read()).decode()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can verify identification documents"),
        (
            "human",
            [
                {"type": "text", "text": "Verify the identification details"},
                {"type": "text", "text": "Name: {user_name}"},
                {"type": "text", "text": "DOB: {user_DOB}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

st.title('KYC Verification Application')

chain =prompt | llm 

st.write('Upload your document')

upload_file=st.file_uploader('upload your file',type=['jpg','png'])

Name=st.text_input("Enter your Name:")
dob=st.text_input("select your DOB")
if upload_file and Name and dob:
    image = encode_image(upload_file)
    response= chain.invoke({'user_name': Name,'user_DOB':dob ,'image':image})
    st.write(response.content)