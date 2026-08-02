import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np
load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
llm=OpenAIEmbeddings(api_key=api_key)

text1=input("Enter your text1")
text2=input("Enter your text2")

response1=llm.embed_query(text1)
response2=llm.embed_query(text2)

similarity_score= np.dot(response1,response2) # cosine similarity

print(similarity_score*100,'%')