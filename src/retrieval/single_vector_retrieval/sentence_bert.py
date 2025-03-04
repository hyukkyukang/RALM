import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample text corpus
texts = [
    "I love programming with Python.",
    "Machine learning is fascinating.",
    "Artificial intelligence will change the world.",
    "Deep learning is a subset of machine learning.",
    "I enjoy solving coding challenges."
]

# Convert texts into embeddings
embeddings = model.encode(texts, normalize_embeddings=True)

# Get the dimensionality of embeddings
d = embeddings.shape[1]

# Create a FAISS index (L2-based index)
index = faiss.IndexFlatL2(d)  # Brute-force search
index.add(embeddings)  # Add the embeddings to the index

# Query text
query_text = "AI is transforming technology."

# Convert query text to an embedding
query_embedding = model.encode([query_text], normalize_embeddings=True)

# Search for the most similar embeddings
k = 2  # Number of closest matches
distances, indices = index.search(query_embedding, k)

# Print results
print(f"Query: {query_text}\n")
for i, idx in enumerate(indices[0]):
    print(f"Match {i+1}: {texts[idx]} (Distance: {distances[0][i]:.4f})")

