import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.globals import set_debug

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model ='gpt-4o',api_key=api_key)


prompt=ChatPromptTemplate.from_messages ([
                        ('system','you are a ajile coach. answer any questions related to ajile process'),
                        MessagesPlaceholder(variable_name='chat_history'),
                        ('human','{input}')
                    ])

chain= prompt | llm
history_of_messages=ChatMessageHistory()

chain_with_history=RunnableWithMessageHistory(
                    chain,
                    lambda session_id: history_of_messages,
                    input_messages_key='input',
                    history_messages_key='chat_history'
                )

while True:
    user_input=input('enter a question:')

    if user_input:
        response=chain_with_history.invoke({'input':user_input},{'configurable':{'session_id':'123'}})
        print(response.content)

