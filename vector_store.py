from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
import pickle
import os

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []  # Store original text chunks
        self.metadata = []   # Store metadata (filename, etc.)
    
    def add_documents(self, chunks: List[str], filename: str):
        """Add document chunks to vector store"""
        try:
            # Generate embeddings
            embeddings = self.model.encode(chunks)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings.astype('float32'))
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store documents and metadata
            self.documents.extend(chunks)
            self.metadata.extend([{"filename": filename, "chunk_id": i} for i in range(len(chunks))])
            
            print(f"Added {len(chunks)} chunks from {filename}")
            
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            if len(self.documents) == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Return results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        "text": self.documents[idx],
                        "score": float(score),
                        "metadata": self.metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching: {e}")
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
            print(f"Error saving index: {e}")
    
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
            print(f"Error loading index: {e}")