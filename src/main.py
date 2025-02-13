import faiss
import numpy as np
from similarity import search_similar_trials, create_index
from utils import process_query
from tqdm import tqdm
from config import queries, embedding_columns # queries and columns on which we're working.
from data import embedding_df
import os

def main():
    # embedding matrix from the embedding_df.
    embedding_matrices = []
    for col in tqdm(embedding_columns, desc="Merging columns' embedding"):  # 4 columns for now.
        col_data = np.vstack(embedding_df[col].values)
        embedding_matrices.append(col_data)

    concatenated_embeddings = np.hstack(embedding_matrices)
    index = create_index(concatenated_embeddings)
    print("Index Created")
    
    # Query Time:
    for query_id in tqdm(queries, desc="Processing Queries"):
        query_embedding= process_query(query_id)
        similar_trials = search_similar_trials(query_embedding, index, k=10)
        output_path = os.path.join("..", "Trial Results", f"{query_id}.csv")
        if os.path.exists(output_path):
            os.remove(output_path)
        similar_trials['nct_id'].to_csv(output_path, index=False)


    from explanability import visualize_embeddings
    visualize_embeddings(concatenated_embeddings, queries, method="tsne")

    # from evaluation import plot_similarity_distribution
    # import matplotlib.pyplot as plt
    # # Plot similarity distribution
    # plot_similarity_distribution(similar_trials)
    # plt.show()

if __name__ == "__main__":
    main()