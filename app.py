from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot using LangGraph, Pinecone and Hugging Face",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


@app.get("/")
async def root():
    return {"message": "Agentic AI RAG API is running"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):

    # Build the RAG graph only when a request arrives.
    from src.graph import build_rag_graph

    graph = build_rag_graph()

    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0,
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"],
    )