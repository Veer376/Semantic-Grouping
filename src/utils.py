import json
import os
import numpy as np
from config import *
from embeddings import get_embedding
# from preprocessing import chunk_text, clean_text
from data import embedding_df

def process_column(df, column_name, chunking_threshold=chunking_threshold, text_type=None):
    if text_type == "embedding":
        df[column_name + '_chunked'] = df[column_name].apply(
            lambda x: chunk_text(x, chunk_size, overlap) if len(x) > chunking_threshold else [x])

        df[column_name + '_embedding'] = df[column_name + '_chunked'].apply(
            lambda chunks: np.mean(get_embedding(chunks), axis=0) if isinstance(chunks, list) and chunks else get_embedding("")
        )
    elif text_type == "tfidf":
        df[column_name + '_cleaned'] = df[column_name].apply(clean_text)
        tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_name + '_cleaned'])
        svd = TruncatedSVD(n_components=svd_n_components)
        reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)
        df[column_name + "_tfidf"] = list(reduced_tfidf_matrix)

    output_file_path = os.path.join(script_dir, '..', 'data', f'{column_name + f"_{text_type}"}.csv')
    df[['nct_id', f'{column_name + f"_{text_type}"}']].to_csv(output_file_path, index=False)

    return df

EMBEDDING_DIM = embedding_dim
def compute_embedding(text_value, chunk_text_fn, get_embedding_fn, chunk_threshold=300, chunk_size=200, overlap=50):
    """
    Computes an embedding by:
      1) Checking the text length against chunk_threshold.
      2) Splitting the text into overlap chunks if it's longer than the threshold.
      3) Getting embeddings for each chunk (via get_embedding_fn) and then averaging them.

    Args:
        text_value (str): The text for which we want to compute embeddings.
        chunk_text_fn (callable): A function that takes (text, size, overlap) and returns a list of text chunks.
        get_embedding_fn (callable): A function that takes a string and returns a NumPy embedding vector.
        chunk_threshold (int, optional): Minimum text length to start chunking. Defaults to 300.
        chunk_size (int, optional): Maximum characters per chunk. Defaults to 200.
        overlap (int, optional): Number of overlapping characters between chunks. Defaults to 50.

    Returns:
        np.ndarray: A float32 NumPy array containing the averaged embedding of dimension EMBEDDING_DIM.
    """
    # If text is missing or not a string, return a zero vector
    if not isinstance(text_value, str) or not text_value:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # If text is longer than the threshold, chunk it
    if len(text_value) > chunk_threshold:
        text_chunks = chunk_text_fn(text_value, chunk_size, overlap)
    else:
        text_chunks = [text_value]

    # Get embeddings for each chunk and then average them
    if text_chunks:
        chunk_embeddings = [get_embedding_fn(chunk) for chunk in text_chunks]
        chunk_embeddings = [emb for emb in chunk_embeddings if emb is not None]  # filter out any None
        if chunk_embeddings:
            # Convert each embedding to float32 if needed, then average
            chunk_arrays = [emb.astype(np.float32, copy=False) for emb in chunk_embeddings]
            mean_embedding = np.mean(chunk_arrays, axis=0)
            return np.ascontiguousarray(mean_embedding)

    # If no valid chunks or embeddings, return zero vector
    return np.zeros(EMBEDDING_DIM, dtype=np.float32)

def process_query(query_id):
    query_row = embedding_df[embedding_df['nct_id'] == query_id].iloc[0]
    return np.hstack([query_row[col] for col in embedding_columns])