import base64
import os 
from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools

#LLM -setup
load_dotenv()
api_key= os.getenv('OPENAI_API_KEY')
llm= ChatOpenAI(model='gpt-4o-mini', api_key=api_key)

#Tools 
tools=load_tools(['wikipedia','ddg-search'])

# ------------------------------
# 3. ReAct-style system prompt
# ------------------------------
react_system_prompt = """
You are a ReAct-style AI agent.

Follow this loop carefully:
1. THOUGHT: Think step by step about what to do next.
2. ACTION: When needed, call one of the tools (wikipedia, ddg-search).
3. OBSERVATION: Read the tool result and decide the next step.

Repeat THOUGHT → ACTION → OBSERVATION
until you are ready to give the final answer.

When you are confident, stop using tools and respond with a clear, concise final answer to the user.
"""


#Create agent

agent= create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt
)


def encode_image(upload_file):
    return base64.b64encode(upload_file.read()).decode()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can identify a landmark."),
        (
            "human",
            [
                {"type": "text", "text": "return the landmark name"},
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

st.title('Landmark-Identifier')

chain =prompt | llm 
upload_file=st.file_uploader('upload your file',type=['jpg','png'])
question=st.text_input("Enter a question about the landmark")

if upload_file and question:
    image = encode_image(upload_file)
    vision_response= chain.invoke({'input': question,'image':image})
    landmark_name=question+' '+vision_response.content

    result = agent.invoke(
        {
            "messages":[{'role':'user',"content":landmark_name + " without explanation"}]
        }
    )
    final_msg=result["messages"][-1]
    st.write(final_msg.content)