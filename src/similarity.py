import numpy as np
from data import df
import faiss


def create_index(concatenated_embeddings):
    faiss.normalize_L2(concatenated_embeddings)
    final_dimention = concatenated_embeddings.shape[1]   # 748 * 4 columns.

    print('Creating FAISS INDEX...')
    index = faiss.IndexFlatIP(final_dimention)
    index.add(concatenated_embeddings)
    return index

def search_similar_trials(query_embedding, index, k=10):
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    similar_trials = df.iloc[indices[0]].copy()  # Create a copy to modify
    similar_trials.loc[:, 'similarity_score'] = distances[0] # No need to do the 1 - distances as we are using the inner product

    return similar_trials

def search_similar_trials2(query_embedding, embeddings_matrix, k=10):
    # Reshape query
    query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings_matrix)
    
    # Get top k indices
    top_indices = np.argsort(similarities[0])[-k:][::-1]
    
    # Get corresponding trials
    similar_trials = df.iloc[top_indices]
    similar_trials.loc[:, 'similarity_score'] = 1 - distances[0]
    
    return similar_trials

def cosine_similarity(query_vector, embedding_matrix):
    """
    Calculate cosine similarity between query vector and embedding matrix
    """
    # Normalize vectors
    query_norm = np.linalg.norm(query_vector)
    matrix_norm = np.linalg.norm(embedding_matrix, axis=1)
    
    # Avoid division by zero
    query_norm = np.maximum(query_norm, np.finfo(float).eps)
    matrix_norm = np.maximum(matrix_norm, np.finfo(float).eps)
    
    # Calculate normalized dot product
    normalized_query = query_vector / query_norm
    normalized_matrix = embedding_matrix / matrix_norm[:, np.newaxis]
    similarities = np.dot(normalized_matrix, normalized_query.T)
    
    return similarities