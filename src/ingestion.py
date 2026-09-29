from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_DIMENSION = 384



def create_pinecone_index():
    """
    Create the Pinecone index if it does not already exist.
    """

    print("Connecting to Pinecone...")

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing_indexes:

        print(
            f"Creating Pinecone index: "
            f"{PINECONE_INDEX_NAME}"
        )

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Pinecone index created successfully.")

    else:

        print(
            f"Pinecone index already exists: "
            f"{PINECONE_INDEX_NAME}"
        )



def load_and_split_pdf(pdf_path: str):
    """
    Load the PDF document and split it into smaller chunks.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():

        raise FileNotFoundError(
            f"PDF file not found: {pdf_file}"
        )

    print("Loading PDF...")

    loader = PyPDFLoader(
        str(pdf_file)
    )

    documents = loader.load()

    print(
        f"Loaded {len(documents)} pages."
    )

    print("Splitting document into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


def create_embeddings():
    """
    Create the local Hugging Face embedding model.

    The model runs locally, so we do not need
    OpenAI API credits for document embeddings.
    """

    print(
        "Loading Hugging Face embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print(
        "Hugging Face embedding model loaded."
    )

    return embeddings



def run_ingestion(pdf_path: str):
    """
    Complete RAG ingestion pipeline:

    PDF
      ↓
    Text Extraction
      ↓
    Text Chunking
      ↓
    Hugging Face Embeddings
      ↓
    Pinecone Vector Database
    """

    print("\n")
    print("=" * 60)
    print("STARTING DOCUMENT INGESTION")
    print("=" * 60)

    # Step 1: Create/check Pinecone index
    create_pinecone_index()

    # Step 2: Load and split PDF
    chunks = load_and_split_pdf(
        pdf_path
    )

    # Step 3: Create local embeddings
    embeddings = create_embeddings()

    # Step 4: Upload vectors to Pinecone
    print(
        "Uploading document chunks to Pinecone..."
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    print("\n")
    print("=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Total chunks uploaded: {len(chunks)}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        f"Pinecone index: "
        f"{PINECONE_INDEX_NAME}"
    )


if __name__ == "__main__":

    pdf_path = "data/Ebook-Agentic-AI.pdf"

    run_ingestion(
        pdf_path
    )