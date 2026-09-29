from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_rag_graph


# Create FastAPI application
app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot using LangGraph, Pinecone and Hugging Face",
    version="1.0.0",
)

# Build LangGraph

graph = build_rag_graph()


# Request Model

class QueryRequest(BaseModel):
    query: str

# Response Model

class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


# Root Endpoint

@app.get("/")
async def root():

    return {
        "message": "Agentic AI RAG API is running"
    }


# Chat Endpoint

@app.post(
    "/chat",
    response_model=QueryResponse
)
async def chat_endpoint(
    request: QueryRequest
):

    # Initial LangGraph state
    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0,
    }

    # Run LangGraph
    result = graph.invoke(
        initial_state
    )

    # Return API response
    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"],
    )