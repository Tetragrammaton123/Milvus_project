from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
from loguru import logger

def build_milvus_index():
    logger.info("Connecting to Milvus database")
    connections.connect("default", uri="milvus.db")
    logger.success("Connected to Milvus database")

    logger.info("Loading ML-ArXiv-Papers dataset")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")
    df = pd.DataFrame(dataset['train'])[['title', 'abstract']].dropna()
    df['text'] = df['title'] + '. ' + df['abstract']
    logger.success(f"Dataset loaded with {len(df)} papers")

    model = SentenceTransformer("all-MiniLM-L6-v2", device=None)  # 'None' -> detect cuda/mps/cpu automatically
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
    collection = Collection("ml_arxiv", schema)
    logger.info("Collection 'ml_arxiv' created")

    logger.info("Inserting documents into collection")
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        collection.insert([df['title'].iloc[i:end].tolist(), embeddings[i:end]])
        logger.debug(f"Inserted batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
    logger.success(f"All {len(df)} documents inserted")

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
