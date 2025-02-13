import os
import numpy as np
import pandas as pd
from config import embedding_dir as dir, data_dir
# embeddings_dir = dir
# # Get a list of all CSV files in the directory
# embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith(".csv")]
#
# # Expected embedding dimension for ClinicalBERT
# expected_dim = 768
#
# for file in embedding_files:
#     filepath = os.path.join(embeddings_dir, file)
#     df = pd.read_csv(filepath)
#
#     # Assuming the embedding column is named 'embedding'
#     embedding_column = df['Study Title_embedding']  # Replace 'embedding' with the actual column name if different
#
#     # Check if the column contains lists
#     if embedding_column.apply(lambda x: isinstance(x, list)).all():
#         # Convert the lists to NumPy arrays for shape analysis
#         embedding_column = embedding_column.apply(np.array)
#
#     # Check the dimensions of the embeddings
#     for i, embedding in enumerate(embedding_column):
#         if embedding.shape[0] != expected_dim:
#             print(f"Warning: Inconsistent dimension in {file}, row {i}: "
#             f"Expected {expected_dim}, found {embedding.shape[0]}")
#
#


# from src.embeddings import get_embedding
# from utils import process_column
# from data import df
# new_df = df.iloc[:5]
# print('processing')
# process_column(new_df, "Study_Title", text_type="embedding")
#
# embedding = pd.read_csv(os.path.join(data_dir, 'Study_Title_embedding.csv'))
# list= embedding['Study_Title_embedding'][1]
# print(list)
# print(type(list))
# print(len(list))
import os
import numpy as np
import pandas as pd

# embedding_df = df[['nct_id']]
def load_embeddings_from_csv(filepath, column_name):
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if column_name in df.columns:
                embeddings = []
                for idx, embedding_str in enumerate(df[column_name]):
                    # Remove '[' and ']' if they exist
                    embedding_str = str(embedding_str).replace('[', '').replace(']', '')
                    # Remove line breaks or extra whitespace
                    embedding_str = embedding_str.replace('\n', ' ').strip()
                    # Split on whitespace rather than commas (handles multiline floats)
                    # If your file uses commas consistently, you can revert to .split(',')
                    values = embedding_str.split()
                    # Convert each split token to a float
                    embedding = np.array([np.float32(val) for val in values], dtype=np.float32)
                    embeddings.append(embedding)

                # Stack individual embedding arrays into a 2D array
                embeddings = np.vstack(embeddings)
                # embedding_df[column_name] = embeddings
                print(f"Embeddings loaded from '{filepath}' with shape: {embeddings.shape}")
                return embeddings
            else:
                print(f"Warning: Column '{column_name}' not found in {filepath}.")
                return None

        except Exception as e:
            print(f"Error loading or processing embeddings from {filepath}: {e}")
            return None
    else:
        print(f"Warning: File not found: {filepath}")
        return None

# Example usage (adjust paths as needed):
# embedding_dir = "path_to_embedding_directory"
# Study_Title_embedding = load_embeddings_from_csv(os.path.join(embedding_dir, "Study_Title_embedding.csv"), column_name="Study Title_embedding")
# embedding_df['Study_Title_embedding'] = list(Study_Title_embedding)
# print(embedding_df.info())
#
# criteria_embedding = load_embeddings_from_csv(os.path.join(embedding_dir, "criteria_embedding.csv"), column_name="criteria_embedding")
# embedding_df['Criteria_embedding'] = list(criteria_embedding)
# print(embedding_df.info())
#
# Brief_Summary_embedding = load_embeddings_from_csv(os.path.join(embedding_dir, "Brief_Summary_embedding.csv"), column_name= "Brief Summary_embedding")
# embedding_df['Brief_Summary_embedding'] = list(Brief_Summary_embedding)
# print(embedding_df.info())
#
# Primary_Outcome_Measures_embedding = load_embeddings_from_csv(os.path.join(embedding_dir, "Primary_Outcome_Measures_embedding.csv"), column_name="Primary Outcome Measures_embedding")
# embedding_df['Primary_Outcome_Measures_embedding']= list(Primary_Outcome_Measures_embedding)
# print(embedding_df.info())
#
#
# embedding_df.to_parquet(os.path.join(data_dir, "embedding_df.parquet"), engine="pyarrow")
#

# test_df= pd.read_parquet(os.path.join(data_dir, "embedding_df.parquet"), engine="pyarrow")
#
# embedding_column = test_df['Study_Title_embedding'].values  # This is an array of arrays
# embedding_matrix = np.vstack(embedding_column)
#
# print("DataFrame shape:", test_df.shape)           # e.g., (117980, 5)
# print("Embedding matrix shape:", embedding_matrix.shape)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import shap
from typing import Dict, List

class EmbeddingExplainer:
    def __init__(self, embedding_dim=768 * 4):
        self.pca = PCA(n_components=2)
        self.tsne = TSNE(n_components=2, random_state=42)
        self.tfidf = TfidfVectorizer(max_features=100)

    def explain_similarity(self, query_trial: Dict, similar_trial: Dict, section_weights: Dict) -> Dict:
        """
        Explain why trials are similar based on embeddings, considering section importance.
        """
        explanations = {}
        for section in section_weights:
            section_sim = np.dot(query_trial[f"{section}_embedding"],
                                 similar_trial[f"{section}_embedding"])
            explanations[f"{section}_similarity"] = section_sim * section_weights[section]
        return explanations

    def visualize_embedding_space(self, embeddings: np.ndarray, labels: List[str], title: str):
        """
        Visualize embeddings in 2D space using t-SNE, with adjusted perplexity for small datasets.
        """
        n_samples = embeddings.shape[0]
        perplexity = min(5, n_samples - 1)  # Perplexity must be less than n_samples
        
        if n_samples < 2:
            raise ValueError("Not enough samples to perform t-SNE visualization.")
        
        # Initialize t-SNE with dynamic perplexity
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create the scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)

        # Add labels for points
        for i, label in enumerate(labels):
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

    def compare_text_sections(self, query_text: str, similar_text: str) -> Dict:
        """
        Compare text sections using TF-IDF.
        """
        tfidf_matrix = self.tfidf.fit_transform([query_text, similar_text])
        feature_names = self.tfidf.get_feature_names_out()

        query_terms = dict(zip(feature_names, tfidf_matrix[0].toarray()[0]))
        similar_terms = dict(zip(feature_names, tfidf_matrix[1].toarray()[0]))

        return {
            'shared_terms': set(query_terms.keys()) & set(similar_terms.keys()),
            'query_specific': set(query_terms.keys()) - set(similar_terms.keys()),
            'similar_specific': set(similar_terms.keys()) - set(query_terms.keys())
        }

    def plot_section_similarity(self, section_similarities: Dict):
        """
        Plot section-wise similarity contributions as a bar chart.
        """
        plt.figure(figsize=(10, 6))
        sections = list(section_similarities.keys())
        similarities = list(section_similarities.values())
        sns.barplot(x=sections, y=similarities, palette='viridis')
        plt.title('Section-wise Similarity Contribution')
        plt.xticks(rotation=45)
        plt.ylabel('Weighted Similarity Score')
        plt.show()

def evaluate_with_explanation(query_trial: Dict, similar_trials_df: pd.DataFrame,
                              embedding_df: pd.DataFrame, section_weights: Dict):
    """
    Enhanced evaluation with explainability for clinical trials.
    """
    explainer = EmbeddingExplainer()

    # Visualize embedding space
    embeddings = np.vstack([query_trial['embedding']] + similar_trials_df['embedding'].to_list())
    labels = ['Query'] + similar_trials_df['nct_id'].to_list()
    explainer.visualize_embedding_space(embeddings, labels, title='Query and Similar Trials Embedding Space')

    # Get section-wise similarities for top match
    top_match = similar_trials_df.iloc[0]
    section_similarities = explainer.explain_similarity(
        query_trial=query_trial,
        similar_trial=top_match,
        section_weights=section_weights
    )

    # Plot section similarities
    explainer.plot_section_similarity(section_similarities)

    # Compare text sections
    text_comparison = explainer.compare_text_sections(
        query_text=query_trial['Brief_Summary'],
        similar_text=top_match['Brief_Summary']
    )

    return {
        'top_match_nct_id': top_match['nct_id'],
        'section_similarities': section_similarities,
        'text_comparison': text_comparison
    }

def shap_explain_embedding(query_embedding: np.ndarray, similar_embeddings: np.ndarray):
    """
    Explain similarity using SHAP values for embedding features.
    """
    # SHAP Explanation
    background = similar_embeddings[:50]  # Background samples for SHAP
    explainer = shap.KernelExplainer(lambda x: np.dot(x, query_embedding), background)
    shap_values = explainer.shap_values(similar_embeddings)

    # Visualization
    shap.summary_plot(shap_values, similar_embeddings, feature_names=[f"Dim {i}" for i in range(similar_embeddings.shape[1])])

# Example usage
if True:
    # Example query and similar trials embeddings (random data for demo purposes)
    query_trial = {
        'nct_id': 'NCT00385736',
        'embedding': np.random.rand(768 * 4),
        'Brief_Summary': "This is an example summary of the query trial.",
        'Study_Title_embedding': np.random.rand(768),
        'Brief_Summary_embedding': np.random.rand(768),
        'Primary_Outcome_Measures_embedding': np.random.rand(768),
        'Criteria_embedding': np.random.rand(768),
    }
    
    similar_trials_df = pd.DataFrame({
        'nct_id': ['NCT00386607', 'NCT03518073'],
        'embedding': [np.random.rand(768 * 4), np.random.rand(768 * 4)],
        'Brief_Summary': [
            "Example summary of similar trial 1.",
            "Example summary of similar trial 2."
        ],
        'Study_Title_embedding': [np.random.rand(768), np.random.rand(768)],
        'Brief_Summary_embedding': [np.random.rand(768), np.random.rand(768)],
        'Primary_Outcome_Measures_embedding': [np.random.rand(768), np.random.rand(768)],
        'Criteria_embedding': [np.random.rand(768), np.random.rand(768)],
    })

    section_weights = {
        'Study_Title': 0.3,
        'Brief_Summary': 0.25,
        'Primary_Outcome_Measures': 0.25,
        'Criteria': 0.2
    }

    explanation = evaluate_with_explanation(
        query_trial=query_trial,
        similar_trials_df=similar_trials_df,
        embedding_df=similar_trials_df,
        section_weights=section_weights
    )

    print("Explainability Results:", explanation)

    # SHAP explanation for embeddings
    shap_explain_embedding(query_trial['embedding'], np.vstack(similar_trials_df['embedding']))
