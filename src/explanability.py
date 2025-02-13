import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(embeddings: np.ndarray, labels: list, method="tsne"):
    """
    Visualize high-dimensional embeddings in 2D space.

    Args:
        embeddings (np.ndarray): The high-dimensional embeddings (shape: [num_samples, embedding_dim]).
        labels (list): List of labels corresponding to each embedding.
        method (str): Dimensionality reduction method ('tsne' or 'pca').

    Returns:
        None: Displays a 2D scatter plot of the embeddings.
    """
    # Step 1: Reduce dimensionality (PCA -> t-SNE)
    print("Reducing dimensions...")
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"Explained variance by PCA: {np.sum(pca.explained_variance_ratio_):.2f}")

    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_pca)
    elif method == "pca":
        embeddings_2d = PCA(n_components=2).fit_transform(embeddings_pca)
    else: 
        raise ValueError("Method must be 'tsne' or 'pca'")

    # Step 2: Plot the embeddings in 2D
    print("Plotting embeddings...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.1)

    # Annotate some points with labels
    for i, label in enumerate(labels):
        # if i % 10 == 0:  # Show labels for every 10th point to avoid clutter
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

    plt.title(f"2D Visualization of Embeddings ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
    print('Finish!')
