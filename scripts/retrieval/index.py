import faiss
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# Load ModernBERT model and tokenizer
model_name = "joe32140/ModernBERT-base-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to encode text into embeddings
def get_embedding(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token representation

# Function to process and save embeddings to disk
def save_embeddings(texts, embedding_file="embeddings.npy", text_file="texts.npy"):
    embeddings = np.array([get_embedding(text) for text in texts])  # Convert to NumPy
    np.save(embedding_file, embeddings)
    np.save(text_file, np.array(texts))
    print(f"Saved embeddings to {embedding_file} and texts to {text_file}")

# Function to load embeddings from disk
def load_embeddings(embedding_file="embeddings.npy", text_file="texts.npy"):
    if Path(embedding_file).exists() and Path(text_file).exists():
        embeddings = np.load(embedding_file)
        texts = np.load(text_file, allow_pickle=True)
        print(f"Loaded {len(texts)} embeddings from disk.")
        return embeddings, texts
    else:
        print("No embeddings found. Please generate them first.")
        return None, None

# Function to cluster embeddings using FAISS
def create_faiss_index(embedding_file="embeddings.npy", num_clusters=100):
    embeddings, texts = load_embeddings(embedding_file)
    if embeddings is None:
        return None

    d = embeddings.shape[1]  # Embedding dimension
    quantizer = faiss.IndexFlatL2(d)  # Base index
    index = faiss.IndexIVFFlat(quantizer, d, num_clusters)  # Inverted File Index

    # Train the FAISS index
    index.train(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, "faiss_index.bin")  # Save index to disk
    print(f"FAISS index created with {num_clusters} clusters and saved to faiss_index.bin.")
    return index

# Function to search FAISS index
def search_faiss(query_text, top_k=3, embedding_file="embeddings.npy", text_file="texts.npy", index_file="faiss_index.bin"):
    index = faiss.read_index(index_file) if Path(index_file).exists() else None
    if index is None:
        print("FAISS index not found. Please create it first.")
        return

    query_embedding = get_embedding([query_text])  # Convert query text to an embedding
    distances, indices = index.search(query_embedding, top_k)

    # Load original texts
    _, texts = load_embeddings(embedding_file, text_file)

    print(f"Query: {query_text}\n")
    for i, idx in enumerate(indices[0]):
        print(f"Match {i+1}: {texts[idx]} (Distance: {distances[0][i]:.4f})")

# Example Usage:
if __name__ == "__main__":
    # Sample large text corpus
    texts = [
        "I love programming with Python.",
        "Machine learning is fascinating.",
        "Artificial intelligence will change the world.",
        "Deep learning is a subset of machine learning.",
        "I enjoy solving coding challenges.",
        "Neural networks power modern AI."
    ]

    # Save embeddings
    save_embeddings(texts)

    # Create FAISS index with clustering
    create_faiss_index()

    # Perform search
    search_faiss("AI is transforming technology.", top_k=2)