import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
llm=OpenAIEmbeddings(api_key=api_key)

text=['I love playing video games',
        'I am going to the movie',
        'I love coding',
        'Hello World!']
response=llm.embed_documents(text)
print(len(response))
print(response[0])