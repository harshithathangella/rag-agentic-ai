import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in the .env file")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the .env file")