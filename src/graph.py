from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from openai import OpenAI

from src.config import (
    HF_TOKEN,
    PINECONE_INDEX_NAME,
)


# Configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "openai/gpt-oss-120b"

TOP_K = 6

# Minimum similarity required to consider retrieved
# content sufficiently relevant.
RELEVANCE_THRESHOLD = 0.35


# LangGraph State

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


# Hugging Face LLM Client

llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


# Build RAG Graph

def build_rag_graph():

    # Local embedding model

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    # Pinecone vector store

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    # Retrieval Node

    def retrieve_node(state: AgentState):

        question = state["question"]

        print("\nRetrieving relevant document chunks...")

        # Retrieve documents WITH similarity scores
        results = vectorstore.similarity_search_with_score(
            question,
            k=TOP_K,
        )

        print(
            f"Retrieved {len(results)} chunks."
        )

        context = []

        scores = []

        for document, similarity_score in results:

            print(
                f"Similarity score: "
                f"{similarity_score:.4f}"
            )

            context.append(
                document.page_content
            )

            scores.append(
                float(similarity_score)
            )

        # Calculate relevance score

        if scores:

            # Average similarity of retrieved chunks
            relevance_score = sum(scores) / len(scores)

        else:

            relevance_score = 0.0

        # Keep score between 0 and 1
        relevance_score = max(
            0.0,
            min(1.0, relevance_score)
        )

        print(
            f"Average relevance score: "
            f"{relevance_score:.4f}"
        )

        return {
            "context": context,
            "score": relevance_score,
        }

    # Generation Node

    def generate_node(state: AgentState):

        question = state["question"]

        context = state["context"]

        score = state["score"]

        print(
            "Generating answer using Hugging Face..."
        )

        # No sufficiently relevant information

        if not context or score < RELEVANCE_THRESHOLD:

            return {
                "answer": (
                    "I cannot answer this question "
                    "based on the provided document."
                )
            }

        # Prepare context

        context_text = "\n\n---\n\n".join(
            context
        )

        # Strict RAG prompt

        prompt = f"""
You are a strict document-grounded RAG assistant.

Answer the user's question using ONLY the information
provided in the document context below.

Rules:

1. Do not use outside knowledge.
2. Do not use your general knowledge.
3. Do not invent facts.
4. Do not make assumptions that are not supported
   by the context.
5. If the context does not contain enough information,
   respond exactly:

"I cannot answer this question based on the provided document."

6. Give a concise and clear answer.

Document Context:
-----------------

{context_text}

-----------------

User Question:
{question}
"""

        # Call Hugging Face

        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=500,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer
        }

    # Create LangGraph

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    # START → RETRIEVE
    workflow.add_edge(
        START,
        "retrieve"
    )

    # RETRIEVE → GENERATE
    workflow.add_edge(
        "retrieve",
        "generate"
    )

    # GENERATE → END
    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()