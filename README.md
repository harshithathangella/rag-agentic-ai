# Agentic AI RAG Chatbot

A document-grounded Retrieval-Augmented Generation (RAG) chatbot built using **Python, LangGraph, Pinecone, Hugging Face, and FastAPI**.

The chatbot answers questions using information retrieved from the provided Agentic AI eBook. If the requested information is not available in the document, the system is designed to refuse the question rather than rely on general knowledge.

## Features

* PDF document ingestion
* Text chunking with overlapping chunks
* Local semantic embeddings using Sentence Transformers
* Pinecone vector database for document retrieval
* LangGraph-based RAG workflow
* Hugging Face LLM for answer generation
* Strict document-grounded responses
* Retrieval relevance score
* FastAPI REST API
* Swagger API documentation
* Automated pytest test suite
* Environment-based configuration for API keys

## Architecture

```text
                    Agentic AI eBook
                           |
                           v
                     PDF Loader
                           |
                           v
                    Text Chunking
                           |
                           v
             Sentence Transformer
                  Embeddings
                           |
                           v
                       Pinecone
                    Vector Store
                           |
                           v
                       LangGraph
                    RAG Workflow
                           |
              +------------+------------+
              |                         |
              v                         v
       Retrieve Relevant         Relevance Check
          Chunks                       |
              |                         |
              +------------+------------+
                           |
                           v
                    Hugging Face LLM
                           |
                           v
                   Grounded Answer
                           |
                           v
                       FastAPI
```

## Tech Stack

| Technology            | Purpose                           |
| --------------------- | --------------------------------- |
| Python                | Application development           |
| LangGraph             | RAG workflow orchestration        |
| LangChain             | Document and retrieval components |
| Sentence Transformers | Local text embeddings             |
| Hugging Face          | LLM inference                     |
| Pinecone              | Vector database                   |
| FastAPI               | REST API                          |
| PyPDF                 | PDF document loading              |
| Pytest                | Automated testing                 |

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   └── ingestion.py
│
├── tests/
│   └── test_rag.py
│
├── .env
├── .env.example
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── test_graph.py
```

> `test_graph.py` is a temporary development script and can be removed before the final submission. Automated tests are maintained in `tests/test_rag.py`.

## Prerequisites

* Python 3.10+
* A Pinecone account and API key
* A Hugging Face account and access token with inference permissions

## Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/harshithathangella/rag-agentic-ai
cd rag-agentic-ai
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

Do not commit the `.env` file to GitHub.

A safe template is provided in `.env.example`.

## Document Ingestion

The eBook is located at:

```text
data/Ebook-Agentic-AI.pdf
```

Run the ingestion script from the project root:

```powershell
python -m src.ingestion
```

The ingestion process:

1. Loads the PDF.
2. Splits the document into overlapping text chunks.
3. Generates 384-dimensional embeddings using `all-MiniLM-L6-v2`.
4. Creates the Pinecone index if it does not already exist.
5. Uploads the document chunks and embeddings to Pinecone.

## Run the API

Start the FastAPI application:

```powershell
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoint

### POST `/chat`

Request:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Example response:

```json
{
  "final_answer": "Agentic AI is described as a form of artificial intelligence that enables autonomous decision-making and action.",
  "retrieved_context": [
    "Relevant content retrieved from the Agentic AI eBook..."
  ],
  "confidence_score": 0.72
}
```

The `confidence_score` represents the **retrieval relevance score** used by the RAG pipeline. It should not be interpreted as a calibrated probability that the generated answer is correct.

## Grounding and Refusal Behavior

The chatbot is designed to answer only from information retrieved from the Agentic AI eBook.

For example:

```text
Question:
What is Agentic AI according to the eBook?

Result:
Answered using retrieved eBook content.
```

For an unrelated question:

```text
Question:
Who won the 2022 FIFA World Cup?

Result:
The chatbot refuses to answer because the information is not available in the provided document.
```

This helps reduce unsupported answers and keeps the chatbot grounded in the supplied knowledge source.

## LangGraph Workflow

The RAG workflow follows these main steps:

```text
User Question
      |
      v
Retrieve relevant chunks from Pinecone
      |
      v
Calculate retrieval relevance
      |
      v
Check whether sufficient relevant context exists
      |
      +------ No ------> Refuse to answer
      |
     Yes
      |
      v
Generate answer using Hugging Face LLM
      |
      v
Return answer + context + relevance score
```

## Benchmark Queries

The following queries were used to validate the system:

| Query                                                        | Expected Behavior                                  |
| ------------------------------------------------------------ | -------------------------------------------------- |
| What is Agentic AI according to the eBook?                   | Answer from document                               |
| How do AI agents differ from traditional automation systems? | Answer from document                               |
| What are the core components of an Agentic Architecture?     | Answer from document                               |
| What role does memory play in Agentic AI workflows?          | Answer from document                               |
| Who won the 2022 FIFA World Cup?                             | Refuse because information is outside the document |

## Testing

The project includes automated tests using pytest.

Run:

```powershell
python -m pytest -v
```

The test suite validates:

* Agentic AI retrieval
* Traditional automation comparison
* Agentic Architecture retrieval
* Memory-related retrieval
* Out-of-document question refusal

Current validation:

```text
5 tests passed
```

## Security

API keys and tokens are loaded from environment variables.

The following files and directories are excluded from Git:

```text
.env
venv/
.venv/
__pycache__/
.pytest_cache/
.vscode/
```

Never commit API keys, access tokens, passwords, or other secrets to the repository.

## Future Improvements

* Add a Streamlit user interface
* Add source/page references to retrieved chunks
* Improve retrieval using reranking
* Add more comprehensive evaluation metrics
* Add conversation memory
* Add Docker support
* Add CI testing with GitHub Actions

## License

This project was created as an AI/RAG technical assignment.
