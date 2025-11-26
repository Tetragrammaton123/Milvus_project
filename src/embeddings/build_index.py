from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
from loguru import logger
from tqdm import tqdm

def build_milvus_index():
    logger.info("Connecting to Milvus database")
    connections.connect("default", uri="milvus.db")
    logger.success("Connected to Milvus database")

    logger.info("Loading ML-ArXiv-Papers dataset")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")
    df = pd.DataFrame(dataset['train'])[['title', 'abstract']].dropna()
    
    # Truncate fields to fit within Milvus schema constraints
    MAX_TITLE_LENGTH = 512
    MAX_ABSTRACT_LENGTH = 2048
    
    title_truncated = (df['title'].str.len() > MAX_TITLE_LENGTH).sum()
    abstract_truncated = (df['abstract'].str.len() > MAX_ABSTRACT_LENGTH).sum()
    
    if title_truncated > 0:
        logger.warning(f"Truncating {title_truncated} titles that exceed {MAX_TITLE_LENGTH} characters")
    if abstract_truncated > 0:
        logger.warning(f"Truncating {abstract_truncated} abstracts that exceed {MAX_ABSTRACT_LENGTH} characters")
    
    df['title'] = df['title'].str[:MAX_TITLE_LENGTH]
    df['abstract'] = df['abstract'].str[:MAX_ABSTRACT_LENGTH]
    df['text'] = df['title'] + '\n\n' + df['abstract']
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
        FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, "Архив статей с эмбеддингами SBERT")
    collection = Collection("ml_arxiv", schema)
    logger.info("Collection 'ml_arxiv' created")

    batch_size = 1000
    for i in tqdm(range(0, len(df), batch_size), desc="Inserting documents", unit="batch"):
        end = min(i + batch_size, len(df))
        collection.insert([
            df['title'].iloc[i:end].tolist(),
            df['abstract'].iloc[i:end].tolist(),
            embeddings[i:end]
        ])

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
