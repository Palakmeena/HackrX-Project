import faiss
import numpy as np
import requests
import os
import pickle
from typing import List, Dict

class VectorStore:
    def __init__(self):
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []  # Store original text chunks
        self.metadata = []   # Store metadata (filename, etc.)
        self.api_token = os.getenv("HF_API_KEY")
        if not self.api_token:
            raise ValueError("HF_API_KEY not found in environment. Make sure to set it in your .env file.")

    def get_embedding(self, text: str) -> List[float]:
        """Call Hugging Face Inference API to get embedding"""
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        payload = {
            "inputs": text
        }
        response = requests.post(
            "https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")

    def add_documents(self, chunks: List[str], filename: str):
        """Add document chunks to vector store"""
        try:
            embeddings = [self.get_embedding(chunk) for chunk in chunks]
            embeddings = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

            self.documents.extend(chunks)
            self.metadata.extend([{"filename": filename, "chunk_id": i} for i in range(len(chunks))])

            print(f"✅ Added {len(chunks)} chunks from {filename}")

        except Exception as e:
            print(f"❌ Error adding documents: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            if not self.documents:
                return []

            query_embedding = self.get_embedding(query)
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)

            scores, indices = self.index.search(query_embedding, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        "text": self.documents[idx],
                        "score": float(score),
                        "metadata": self.metadata[idx]
                    })

            return results

        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []

    def save_index(self, path: str = "vector_store.pkl"):
        """Save vector store to file"""
        try:
            data = {
                "documents": self.documents,
                "metadata": self.metadata
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            faiss.write_index(self.index, "faiss_index.bin")
        except Exception as e:
            print(f"❌ Error saving index: {e}")

    def load_index(self, path: str = "vector_store.pkl"):
        """Load vector store from file"""
        try:
            if os.path.exists(path) and os.path.exists("faiss_index.bin"):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                self.index = faiss.read_index("faiss_index.bin")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
