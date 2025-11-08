from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import math

def build_milvus_index():
    connections.connect("default", uri="milvus.db")

    dataset = load_dataset("CShorten/ML-ArXiv-Papers")
    df = pd.DataFrame(dataset['train'])[['title', 'abstract']].dropna()
    df['text'] = df['title'] + '. ' + df['abstract']

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = model.encode(df['text'].tolist(), convert_to_numpy=True, batch_size=64)

    dim = embeddings.shape[1]
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, "Архив статей с эмбеддингами SBERT")
    collection = Collection("ml_arxiv", schema)

    batch_size = 1000
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        collection.insert([df['title'].iloc[i:end].tolist(), embeddings[i:end]])

    index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

    return collection, model

if __name__ == "__main__":
    build_milvus_index()
