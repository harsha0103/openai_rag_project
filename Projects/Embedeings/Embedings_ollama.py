import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings


llm=OllamaEmbeddings(model='mistral')

text=input("Enter your text")
response=llm.embed_query(text)
print(response)