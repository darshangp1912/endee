import requests
from sentence_transformers import SentenceTransformer
from endee import Endee
import config


class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        print(f"Connecting to Endee at {config.ENDEE_BASE_URL} ...")
        client = Endee(token=None)
        client.set_base_url(config.ENDEE_BASE_URL)
        self.index = client.get_index(config.INDEX_NAME)

        self.ollama_url = f"{config.OLLAMA_BASE_URL}/api/generate"
        self.ollama_model = config.OLLAMA_MODEL

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Embed the query and find the top-k most similar chunks in Endee."""
        query_vector = self.model.encode(query).tolist()
        results = self.index.query(vector=query_vector, top_k=top_k)
        # Each result dict has a 'meta' key containing our stored 'document' text
        return [r.get("meta", {}).get("document", "") for r in results]

    def generate_answer(self, query: str, contexts: list[str]) -> str:
        """Send the retrieved contexts + question to Ollama and return the answer."""
        if not contexts:
            return "I could not find any relevant information in the provided documents."

        context_block = "\n\n".join(
            f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        prompt = (
            "You are an AI Interview Assistant. "
            "Use ONLY the context below to answer the question. "
            "If the answer is not in the context, say you don't know based on the provided documents. "
            "Be concise and accurate.\n\n"
            f"{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def ask(self, query: str) -> tuple[str, list[str]]:
        """Full RAG pipeline: retrieve relevant chunks, then generate an answer."""
        contexts = self.retrieve(query)
        answer = self.generate_answer(query, contexts)
        return answer, contexts
