# vector_db.py
import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim=512):
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after norm
        self.metadata = []
        # Keep a copy of normalized vectors so we can visualize them later
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors, labels):
        # Normalize input vectors (avoid divide-by-zero)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors_norm = vectors / norms
        vectors_norm = vectors_norm.astype(np.float32)
        self.index.add(vectors_norm)
        self.metadata.extend(labels)
        # Append to internal storage for visualization and retrieval
        if self._vectors.size == 0:
            self._vectors = vectors_norm.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors_norm])

    def search(self, query, k=3):
            if self.index.ntotal == 0:
                return []  # Empty DB - no results
            query = np.asarray(query)  # Ensure NumPy array
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return []  # Invalid query - avoid div-by-zero or bad results
            query = query / query_norm
            D, I = self.index.search(query.reshape(1, -1), k)
            results = []
            for j, i in enumerate(I[0]):
                if 0 <= i < len(self.metadata):  # Only valid indices
                    results.append((self.metadata[i], D[0][j]))
            return results

    def get_all(self):
        """Return (vectors, labels) for all indexed items.

        Vectors are the normalized vectors stored when added. Returns a
        NumPy array of shape (N, dim) and a list of labels (length N).
        """
        return self._vectors, list(self.metadata)