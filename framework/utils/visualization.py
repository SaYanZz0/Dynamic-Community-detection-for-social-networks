import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import os
import torch

def load_graph_from_file(file_path):
    """Load graph from dataset file"""
    G = nx.Graph()
    
    with open(file_path, 'r') as infile:
        for line in infile:
            parts = line.split()
            s_node = int(parts[0])
            t_node = int(parts[1])
            # Add timestamp as edge attribute if available
            if len(parts) > 2:
                G.add_edge(s_node, t_node, timestamp=float(parts[2]))
            else:
                G.add_edge(s_node, t_node)
    
    return G

def visualize_graph(G, output_dir, title="Graph Visualization", max_nodes=30000):
    """Visualize graph structure with NetworkX"""
    plt.figure(figsize=(12, 10))
    
    # If graph is too large, sample a subgraph
    if len(G.nodes()) > max_nodes:
        print(f"Graph too large ({len(G.nodes())} nodes), sampling {max_nodes} nodes for visualization")
        nodes = list(G.nodes())
        sampled_nodes = np.random.choice(nodes, max_nodes, replace=False)
        G = G.subgraph(sampled_nodes)
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw_networkx(G, pos, 
                    node_size=50, 
                    node_color='lightblue',
                    edge_color='gray',
                    alpha=0.8,
                    with_labels=False)
    
    plt.title(title)
    plt.axis('off')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/graph_structure.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {output_dir}/graph_structure.png")

def load_labels(label_path):
    """Load node labels from file"""
    n2l = dict()
    labels = []
    
    with open(label_path, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            n2l[n_id] = l_id
    
    for i in range(len(n2l)):
        labels.append(int(n2l[i]))
    
    return np.array(labels)

def visualize_embeddings(embeddings, labels, output_dir, title="Node Embeddings Visualization"):
    """Visualize node embeddings using t-SNE"""
    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(12, 10))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                    c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/embeddings_tsne.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Embeddings visualization saved to {output_dir}/embeddings_tsne.png")

def visualize_clustering_metrics(metrics_history, output_dir):
    """Visualize clustering metrics over epochs"""
    epochs = list(range(1, len(metrics_history['acc']) + 1))
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot each metric
    axs[0, 0].plot(epochs, metrics_history['acc'], 'b-')
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    
    axs[0, 1].plot(epochs, metrics_history['nmi'], 'r-')
    axs[0, 1].set_title('NMI')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('NMI')
    
    axs[1, 0].plot(epochs, metrics_history['ari'], 'g-')
    axs[1, 0].set_title('ARI')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('ARI')
    
    axs[1, 1].plot(epochs, metrics_history['f1'], 'y-')
    axs[1, 1].set_title('F1 Score')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('F1 Score')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/metrics_history.png", dpi=300)
    plt.close()
    print(f"Metrics history visualization saved to {output_dir}/metrics_history.png") 