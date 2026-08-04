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
        ("system", "You are a helpful assistant that can describe images."),
        (
            "human",
            [
                {"type": "text", "text": "{input}"},
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

st.title('Image-descriptor')

chain =prompt | llm 
upload_file=st.file_uploader('upload your file',type=['jpg','png'])
question=st.text_input("Enter your question")
if upload_file and question:
    image = encode_image(upload_file)
    response= chain.invoke({'input': question,'image':image})
    st.write(response.content)