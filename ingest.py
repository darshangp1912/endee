import os
from sentence_transformers import SentenceTransformer
from endee import Endee
import config


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 15) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def ingest_documents():
    print("Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"Connecting to Endee at {config.ENDEE_BASE_URL} ...")
    client = Endee(token=None)
    client.set_base_url(config.ENDEE_BASE_URL)

    # Create or reuse the index
    print(f"Creating index '{config.INDEX_NAME}' (ignores error if already exists)...")
    try:
        client.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            space_type="cosine",
        )
        print("  Index created.")
    except Exception as e:
        print(f"  Index already exists or error: {e}")

    index = client.get_index(config.INDEX_NAME)

    # Collect chunks from all .txt files in data/
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.isdir(data_dir):
        print(f"ERROR: data directory not found at '{data_dir}'")
        return

    documents, metadatas = [], []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_id": idx})
        print(f"  '{filename}' -> {len(chunks)} chunks")

    if not documents:
        print("No documents found. Add .txt files to the data/ directory.")
        return

    print(f"\nGenerating embeddings for {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # Build Endee upsert payload
    vectors = [
        {
            "id": f"doc_{i}",
            "vector": embeddings[i],
            "meta": {
                "document": documents[i],
                "source": metadatas[i]["source"],
                "chunk_id": metadatas[i]["chunk_id"],
            },
        }
        for i in range(len(documents))
    ]

    print("Inserting into Endee Vector Database...")
    index.upsert(vectors)
    print(f"\n[SUCCESS] Ingestion complete! Inserted {len(vectors)} chunks into '{config.INDEX_NAME}'.")


if __name__ == "__main__":
    ingest_documents()
