#!/usr/bin/env python3
"""
centroid_selection.py - Utility for selecting instances closest to cluster centroids

This module provides functions to:
1. Load embeddings and centroids
2. Compute distances from instances to their cluster centroids
3. Select k instances closest to the centroid for each cluster

Usage:
    from centroid_selection import select_closest_to_centroid
    
    examples = select_closest_to_centroid(
        cluster_df=df[df['cluster_id'] == 5],
        embeddings=embeddings,
        centroids=centroids,
        cluster_id=5,
        k=5
    )
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.spatial.distance import cdist


def load_embeddings_and_centroids(
    embeddings_path: str = "data/embeddings.npy",
    centroids_path: str = "data/cluster_centroids.npy"
) -> tuple:
    """
    Load embeddings and centroids from saved numpy files.
    
    Args:
        embeddings_path: Path to embeddings.npy
        centroids_path: Path to cluster_centroids.npy
    
    Returns:
        embeddings: (n_instances, embedding_dim) array
        centroids: (n_clusters, embedding_dim) array
    """
    print(f"[INFO] Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"[INFO] Loaded embeddings: shape {embeddings.shape}")
    
    print(f"[INFO] Loading centroids from {centroids_path}...")
    centroids = np.load(centroids_path)
    print(f"[INFO] Loaded centroids: shape {centroids.shape}")
    
    return embeddings, centroids


def compute_distances_to_centroid(
    cluster_embeddings: np.ndarray,
    centroid: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute distances from cluster instances to the cluster centroid.
    
    Args:
        cluster_embeddings: (n_instances_in_cluster, embedding_dim)
        centroid: (embedding_dim,)
        metric: Distance metric ('cosine', 'euclidean', etc.)
    
    Returns:
        distances: (n_instances_in_cluster,) array of distances
    """
    # Reshape centroid to (1, embedding_dim) for cdist
    centroid_reshaped = centroid.reshape(1, -1)
    
    # Compute distances: (n_instances, 1)
    distances = cdist(cluster_embeddings, centroid_reshaped, metric=metric)
    
    # Return as 1D array
    return distances.flatten()


def select_closest_to_centroid(
    cluster_df: pd.DataFrame,
    embeddings: np.ndarray,
    centroids: np.ndarray,
    cluster_id: int,
    k: int = 5,
    prefer_original: bool = True,
    metric: str = 'cosine'
) -> List[Dict]:
    """
    Select k instances closest to the cluster centroid.
    
    Args:
        cluster_df: DataFrame containing all instances in the cluster
        embeddings: All instance embeddings (n_instances, embedding_dim)
        centroids: All cluster centroids (n_clusters, embedding_dim)
        cluster_id: The cluster ID
        k: Number of instances to select
        prefer_original: If True, prefer non-augmented instances
        metric: Distance metric ('cosine', 'euclidean')
    
    Returns:
        List of dicts with selected instances and their info
    """
    if len(cluster_df) == 0:
        return []
    
    # Get centroid for this cluster
    centroid = centroids[cluster_id]
    
    # Get embeddings for instances in this cluster
    # Assumes cluster_df has the same row indices as the original df used to create embeddings
    instance_indices = cluster_df.index.tolist()
    cluster_embeddings = embeddings[instance_indices]
    
    # Compute distances to centroid
    distances = compute_distances_to_centroid(cluster_embeddings, centroid, metric)
    
    # Add distance to dataframe
    cluster_df_with_dist = cluster_df.copy()
    cluster_df_with_dist['distance_to_centroid'] = distances
    
    # Optionally prefer original instances
    if prefer_original and 'is_augmented' in cluster_df_with_dist.columns:
        # Split into original and augmented
        original_df = cluster_df_with_dist[~cluster_df_with_dist['is_augmented']].copy()
        augmented_df = cluster_df_with_dist[cluster_df_with_dist['is_augmented']].copy()
        
        # Sort both by distance
        original_df = original_df.sort_values('distance_to_centroid')
        augmented_df = augmented_df.sort_values('distance_to_centroid')
        
        # Take as many original as we have (up to k)
        if len(original_df) >= k:
            selected = original_df.head(k)
        else:
            # Take all originals + fill with closest augmented
            need_more = k - len(original_df)
            selected = pd.concat([
                original_df,
                augmented_df.head(need_more)
            ])
    else:
        # Just sort by distance and take top k
        cluster_df_with_dist = cluster_df_with_dist.sort_values('distance_to_centroid')
        selected = cluster_df_with_dist.head(k)
    
    # Convert to list of dicts
    examples = []
    for idx, row in selected.iterrows():
        examples.append({
            'predicate_lemma': row.get('predicate_lemma', ''),
            'object_lemma': row.get('object_lemma', ''),
            'sentence': row.get('sentence', ''),
            'instance_id': row.get('instance_id', ''),
            'event_type': row.get('event_type', 'UNKNOWN'),
            'distance_to_centroid': row.get('distance_to_centroid', 0.0),
        })
    
    return examples


def get_centroid_statistics(
    cluster_df: pd.DataFrame,
    embeddings: np.ndarray,
    centroids: np.ndarray,
    cluster_id: int,
    metric: str = 'cosine'
) -> Dict:
    """
    Get statistics about distances to centroid for a cluster.
    
    Args:
        cluster_df: DataFrame containing all instances in the cluster
        embeddings: All instance embeddings
        centroids: All cluster centroids
        cluster_id: The cluster ID
        metric: Distance metric
    
    Returns:
        Dictionary with statistics (mean, std, min, max distances)
    """
    if len(cluster_df) == 0:
        return {}
    
    centroid = centroids[cluster_id]
    instance_indices = cluster_df.index.tolist()
    cluster_embeddings = embeddings[instance_indices]
    
    distances = compute_distances_to_centroid(cluster_embeddings, centroid, metric)
    
    return {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'median_distance': float(np.median(distances)),
        'num_instances': len(distances),
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    ap = argparse.ArgumentParser(description="Test centroid selection")
    ap.add_argument("--clusters", default="data/clusters.csv")
    ap.add_argument("--embeddings", default="data/embeddings.npy")
    ap.add_argument("--centroids", default="data/cluster_centroids.npy")
    ap.add_argument("--cluster-id", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    
    args = ap.parse_args()
    
    # Load data
    print(f"[INFO] Loading clusters from {args.clusters}...")
    df = pd.read_csv(args.clusters)
    
    embeddings, centroids = load_embeddings_and_centroids(args.embeddings, args.centroids)
    
    # Test on one cluster
    cluster_df = df[df['cluster_id'] == args.cluster_id]
    print(f"\n[INFO] Testing cluster {args.cluster_id} with {len(cluster_df)} instances")
    
    # Get statistics
    stats = get_centroid_statistics(cluster_df, embeddings, centroids, args.cluster_id)
    print(f"\n[INFO] Centroid statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Select closest instances
    examples = select_closest_to_centroid(
        cluster_df, embeddings, centroids, args.cluster_id, k=args.k
    )
    
    print(f"\n[INFO] {args.k} instances closest to centroid:")
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Instance ID: {ex['instance_id']}")
        print(f"  Distance: {ex['distance_to_centroid']:.4f}")
        print(f"  Predicate: {ex['predicate_lemma']}")
        print(f"  Object: {ex['object_lemma']}")
        print(f"  Sentence: {ex['sentence'][:80]}...")
