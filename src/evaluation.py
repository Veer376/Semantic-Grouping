import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def evaluate_similarity_distribution(similarity_scores: List[float]) -> Dict:
    """Evaluate distribution of similarity scores"""
    return {
        'mean': np.mean(similarity_scores),
        'std': np.std(similarity_scores),
        'median': np.median(similarity_scores),
        'min': np.min(similarity_scores),
        'max': np.max(similarity_scores)
    }

def plot_similarity_distribution(similar_trials_df, k_values=[1, 3, 5, 10]):
    """Plot similarity score distribution"""
    plt.figure(figsize=(12, 6))
    
    # Plot similarity score distribution
    plt.hist(similar_trials_df['similarity_score'], bins=20, alpha=0.7)
    plt.axvline(similar_trials_df['similarity_score'].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(similar_trials_df['similarity_score'].median(), color='g', linestyle='--', label='Median')
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.grid(True)
    
    # Print statistics
    stats = evaluate_similarity_distribution(similar_trials_df['similarity_score'].values)
    print("\nSimilarity Score Statistics:")
    for metric, value in stats.items():
        print(f"{metric}: {value:.3f}")


    