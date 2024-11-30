import faiss
import sqlite3
import os
import json
import io
import base64
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def query_sqlite(db_path, query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results

faiss_index_id = "faiss_index:v1::test_bank_2"

db_path = os.getenv("SQLITE_STORE_DIR") + "/faiss_store.db"
query = f"select * from kvstore where key = '{faiss_index_id}';"

results = query_sqlite(db_path, query)

print(results[0][0])

value = json.loads(results[0][1])

bytes = io.BytesIO(base64.b64decode(value["faiss_index"]))
index = faiss.deserialize_index(np.loadtxt(bytes, dtype=np.uint8))
test_vector = model.encode("Llama 3.2 3B Instruct").reshape(1, -1)
print(test_vector.shape)

# Perform search with k=5 nearest neighbors
k = 5
distances, indices = index.search(test_vector, k)

print("\nSearch results:")
print(f"Indices of {k} nearest neighbors: {indices[0]}")
print(f"Distances to {k} nearest neighbors: {distances[0]}")

# Print corresponding document IDs
id_by_index = value["id_by_index"] 
print("\nCorresponding document IDs:")
for idx in indices[0]:
    if idx >= 0:  # Valid index
        doc_id = id_by_index[str(idx)]
        print(f"Index {idx} -> Document: {doc_id}")

# Print document chunks
chunk_by_index = value["chunk_by_index"]
print("\nDocument chunks:")
for idx in indices[0]:
    if idx >= 0:  # Valid index
        chunk = json.loads(chunk_by_index[str(idx)])
        print(f"\nIndex {idx}:")
        print(f"Content: {chunk['content']}")
