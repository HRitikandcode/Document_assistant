from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

def create_llm():

    llm = HuggingFaceEndpoint(
        repo_id= "zai-org/GLM-5.3-Flash",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    models = ChatHuggingFace(llm = llm)
    return models







