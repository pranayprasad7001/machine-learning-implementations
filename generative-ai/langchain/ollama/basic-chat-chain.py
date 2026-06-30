# Import Libraries
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load Environment Variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Create Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful AI assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

# Streamlit Framework
st.title("Langchain Demo With Qwen3")
input_text = st.text_input("What question you have in your mind? ")

# Load Ollama Qwen3
llm_model = ChatOllama(model="qwen3:8b")

# Output Parser
output_parser = StrOutputParser()

# Create Chain
chain = prompt | llm_model | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))