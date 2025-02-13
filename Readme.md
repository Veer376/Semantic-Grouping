# Clinical Trial Retrieval and Explainability

This project loads a set of clinical-trial data (stored in df.parquet) and their corresponding embeddings (in embedding_df.parquet). It then performs semantic retrieval to find the most similar trials for a given query, and provides a explainability component to show why certain trials are retrieved.

## Key Components

1. **Data Loading**  
   - [data.py](src/data.py) loads the parquet files (df.parquet and embedding_df.parquet).  
   - These files contain clinical-trial text data (e.g., summaries, criteria) and precomputed embeddings.

2. **Embeddings**  
   - [embeddings.py](src/embeddings.py) defines a function `get_embedding` using a model (e.g., a transformer) to generate numerical representations (embeddings) for text.

3. **Configuration**  
   - [config.py](src/config.py) holds settings such as embedding dimension, chunk size, overlap, and TF-IDF parameters.

4. **Utilities**  
   - [utils.py](src/utils.py) provides functions to process columns for embedding and tf-idf transformations, compute embeddings for queries, etc.

5. **Similarity**  
   - [similarity.py](src/similarity.py) uses FAISS for approximate nearest-neighbor search (or a cosine-similarity–based method) to find trials most relevant to the query embedding.

6. **Explainability**  
   - [explanability.py](src/explanability.py) includes functions to reduce high-dimensional embeddings (PCA, t-SNE) and plot them for visualization.  
   - [test.py](src/test.py) showcases an example “EmbeddingExplainer” class that computes section-level similarity scores and can visualize them, plus a SHAP-based explanation for embeddings.

7. **Evaluation**  
   - [evaluation.py](src/evaluation.py) has helper methods to evaluate and plot similarity score distributions across trials.

8. **Main Script**  
   - [main.py](src/main.py) demonstrates how to bring everything together:  
     1. Merge embeddings from multiple columns.  
     2. Build a FAISS index for efficient semantic retrieval.  
     3. Retrieve the top K most similar trials for each query.  
     4. Optionally visualize embeddings and manage similarity distribution analysis.

## Usage

1. **Data Preparation**  
   Make sure df.parquet and embedding_df.parquet are present in your project’s /data folder.

2. **Run Main**  
   Execute the main script to:  
   - Build the FAISS index of all trial embeddings.  
   - Retrieve the top 10 similar trials for each query listed in config.py.

3. **Explainability**  
   The code in test.py (and references to explanability.py) can visualize why certain trials were retrieved. It can generate t-SNE plots of embeddings and show section-wise similarity scores.

## Future Work

- Implement an actual query-input mechanism so users can type a query instead of looking up by ID.  
- Enhance the explainability logic to provide more detailed SHAP or attention-based insights.  
- Expand chunking and cleaning methods for large text fields.

