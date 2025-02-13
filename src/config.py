import os
chunk_size = 400  # Adjust as needed
overlap = 25  # Adjust as needed
chunking_threshold = 512
embedding_dim = 768  # Dimension of ClinicalBERT embeddings

# TF-IDF parameters
tfidf_max_features = 5000
tfidf_ngram_range = (1, 2)
svd_n_components = 200

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '..', 'data')
embedding_dir = os.path.join(script_dir, '..', 'data', 'processed')

queries = ["NCT00385736", "NCT00386607", "NCT03518073"]
columns=["Study_Title", "Brief_Summary", "Primary_Outcome Measures", "Criteria"]
embedding_columns=["Study_Title_embedding", "Brief_Summary_embedding", "Primary_Outcome_Measures_embedding", "Criteria_embedding"]