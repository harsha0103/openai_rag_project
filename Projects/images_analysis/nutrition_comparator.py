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
        ("system", "You are a helpful assistant that can analyze imnages of nutrition charts"
        " and help chose the right one"),
    (
            "human",
            [
                {"type": "text", "text": "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image1}",
                        "detail": "low",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image2}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

st.title('Image-descriptor')

chain =prompt | llm 
upload_image1=st.file_uploader('upload your first image',type=['jpg','png'])
upload_image2=st.file_uploader('upload your second image',type=['jpg','png'])

question=st.text_input("Enter your question")
if upload_image1 and upload_image2 and question :
    image1 = encode_image(upload_image1)
    image2 = encode_image(upload_image2)
    response= chain.invoke({'input': question,'image1':image1, 'image2':image2})
    st.write(response.content)