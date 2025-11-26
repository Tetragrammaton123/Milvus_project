from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
from loguru import logger
import os
from tqdm import tqdm

def build_milvus_index():
    logger.info("Connecting to Milvus database")
    connections.connect("default", uri="milvus.db")
    logger.success("Connected to Milvus database")

    collection_name = "ml_arxiv"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=None)  # 'None' -> detect cuda/mps/cpu automatically

    # Check if collection already exists
    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists, skipping creation")
        collection = Collection(collection_name)
        logger.debug("Loading existing collection into memory")
        collection.load()
        logger.success("Existing collection loaded and ready for search")
        return collection, model

    # Collection doesn't exist, create it
    logger.info("Loading ML-ArXiv-Papers dataset")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")
    df = pd.DataFrame(dataset['train'])[['title', 'abstract']].dropna()
    df['text'] = df['title'] + '. ' + df['abstract']
    logger.success(f"Dataset loaded with {len(df)} papers")

    logger.debug("Encoding documents into embeddings")
    embeddings = model.encode(df['text'].tolist(), convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    logger.success(f"Generated embeddings with shape {embeddings.shape}")

    logger.debug("Creating collection schema")
    dim = embeddings.shape[1]
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, "Архив статей с эмбеддингами SBERT")
    collection = Collection(collection_name, schema)
    logger.info(f"Collection '{collection_name}' created")

    batch_size = 1000
    for i in tqdm(range(0, len(df), batch_size), desc="Inserting documents", unit="batch"):
        end = min(i + batch_size, len(df))
        collection.insert([df['title'].iloc[i:end].tolist(), embeddings[i:end]])

    logger.info("Creating vector index")
    index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.success("Vector index created")
    
    logger.debug("Loading collection into memory")
    collection.load()
    logger.success("Collection loaded and ready for search")

    return collection, model

if __name__ == "__main__":
    build_milvus_index()
