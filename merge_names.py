#!/usr/bin/env python3


import os
import argparse
import random
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from FlagEmbedding import FlagAutoModel
from tqdm import tqdm


# ---------------- DEFAULT SETTINGS ----------------
SIM_THRESHOLD = 0.85


def load_embedding_model():
    """
    Load BGE embedding model for semantic similarity.
    
    Returns:
        FlagAutoModel instance
    """
    print("[INFO] Loading BGE embedding model...")
    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-base-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=True
    )
    print("[INFO] Model loaded successfully")
    return model


def compute_embeddings(model, texts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    
    Args:
        model: BGE model
        texts: List of text strings
    
    Returns:
        numpy array of embeddings (n_texts, embedding_dim)
    """
    print(f"[INFO] Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts)
    return embeddings


def find_exact_name_matches(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Find clusters with exact name matches.
    
    Args:
        df: DataFrame with cluster_id and cluster_name columns
    
    Returns:
        Dict mapping cluster_name to list of cluster_ids with that name
    """
    name_to_ids = {}
    
    for _, row in df.iterrows():
        cluster_id = int(row['cluster_id'])
        cluster_name = str(row['cluster_name']).strip()
        
        if cluster_name not in name_to_ids:
            name_to_ids[cluster_name] = []
        name_to_ids[cluster_name].append(cluster_id)
    
    # Filter to only names with multiple clusters
    exact_matches = {name: ids for name, ids in name_to_ids.items() if len(ids) > 1}
    
    return exact_matches


def find_semantic_matches(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of clusters with high semantic similarity.
    
    Args:
        df: DataFrame with cluster information
        embeddings: Embeddings for each cluster
        threshold: Similarity threshold (0.0 to 1.0)
    
    Returns:
        List of (cluster_id1, cluster_id2, similarity) tuples
    """
    print(f"[INFO] Finding semantic matches with threshold {threshold}...")
    
    # Compute similarity matrix
    similarity_matrix = embeddings @ embeddings.T
    
    matches = []
    n = len(df)
    
    # Find pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
            sim = similarity_matrix[i, j]
            if sim >= threshold:
                cluster_id1 = int(df.iloc[i]['cluster_id'])
                cluster_id2 = int(df.iloc[j]['cluster_id'])
                matches.append((cluster_id1, cluster_id2, float(sim)))
    
    print(f"[INFO] Found {len(matches)} semantic matches above threshold")
    return matches


def create_merge_groups(
    exact_matches: Dict[str, List[int]],
    semantic_matches: List[Tuple[int, int, float]]
) -> Dict[int, int]:
    """
    Create mapping from original cluster_id to merged cluster_id.
    
    Uses Union-Find algorithm to group clusters that should be merged.
    
    Args:
        exact_matches: Dict of cluster_name -> list of cluster_ids
        semantic_matches: List of (id1, id2, similarity) tuples
    
    Returns:
        Dict mapping cluster_id -> representative cluster_id
    """
    print("[INFO] Creating merge groups...")
    
    # Collect all cluster IDs
    all_ids = set()
    for ids in exact_matches.values():
        all_ids.update(ids)
    for id1, id2, _ in semantic_matches:
        all_ids.add(id1)
        all_ids.add(id2)
    
    # Initialize Union-Find structure
    parent = {cid: cid for cid in all_ids}
    
    def find(x):
        """Find root with path compression"""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        """Union two sets"""
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    
    # Merge exact matches
    for name, ids in exact_matches.items():
        if len(ids) > 1:
            # Merge all IDs with the same name
            for i in range(1, len(ids)):
                union(ids[0], ids[i])
    
    # Merge semantic matches
    for id1, id2, _ in semantic_matches:
        union(id1, id2)
    
    # Build final mapping: cluster_id -> representative
    cluster_to_merged = {}
    for cid in all_ids:
        root = find(cid)
        cluster_to_merged[cid] = root
    
    # Count merge groups
    unique_merged = len(set(cluster_to_merged.values()))
    print(f"[INFO] Created {unique_merged} merged groups from {len(all_ids)} original clusters")
    
    return cluster_to_merged


def assign_merged_names(
    df: pd.DataFrame,
    cluster_to_merged: Dict[int, int],
    seed: int = 42
) -> pd.DataFrame:
    """
    Assign merged cluster IDs and names to all clusters.
    
    For merged groups, randomly picks one cluster's name as the merged name.
    
    Args:
        df: DataFrame with cluster information
        cluster_to_merged: Mapping from cluster_id to merged cluster_id
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with cluster_merged_id and cluster_merged_name columns
    """
    random.seed(seed)
    
    # Group clusters by their merged ID
    merged_to_clusters = {}
    for cid, merged_id in cluster_to_merged.items():
        if merged_id not in merged_to_clusters:
            merged_to_clusters[merged_id] = []
        merged_to_clusters[merged_id].append(cid)
    
    # For each merged group, randomly pick a representative name
    merged_id_to_name = {}
    for merged_id, cluster_ids in merged_to_clusters.items():
        # Get all cluster names in this merged group
        names = []
        for cid in cluster_ids:
            name = df[df['cluster_id'] == cid]['cluster_name'].iloc[0]
            names.append((cid, name))
        
        # Randomly pick one
        chosen_id, chosen_name = random.choice(names)
        merged_id_to_name[merged_id] = (chosen_id, chosen_name)
        
        if len(names) > 1:
            print(f"[INFO] Merged group {merged_id}: {len(names)} clusters -> '{chosen_name}'")
    
    # Create new columns
    df_merged = df.copy()
    
    # Initialize with original values (for clusters that aren't merged)
    df_merged['cluster_merged_id'] = df_merged['cluster_id']
    df_merged['cluster_merged_name'] = df_merged['cluster_name']
    
    # Update merged clusters
    for idx, row in df_merged.iterrows():
        cluster_id = int(row['cluster_id'])
        
        if cluster_id in cluster_to_merged:
            merged_id = cluster_to_merged[cluster_id]
            # Use the chosen merged ID and name
            final_merged_id, final_merged_name = merged_id_to_name[merged_id]
            df_merged.at[idx, 'cluster_merged_id'] = final_merged_id
            df_merged.at[idx, 'cluster_merged_name'] = final_merged_name
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge clusters based on name and semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        '--input',
        default='data/cluster_names_definition.csv',
        help='Input file with cluster names and definitions (default: data/cluster_names_definition.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/cluster_names_merged.csv',
        help='Output file with merged cluster IDs (default: data/cluster_names_merged.csv)'
    )
    
    # Similarity settings
    parser.add_argument(
        '--threshold',
        type=float,
        default=SIM_THRESHOLD,
        help=f'Semantic similarity threshold for merging (default: {SIM_THRESHOLD})'
    )
    
    # System settings
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"[INFO] Reading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} clusters")
    
    # Verify required columns
    required_cols = ['cluster_id', 'cluster_name', 'definition']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Find exact name matches
    print("\n[INFO] Step 1: Finding exact name matches...")
    exact_matches = find_exact_name_matches(df)
    print(f"[INFO] Found {len(exact_matches)} cluster names with duplicates")
    for name, ids in list(exact_matches.items())[:5]:
        print(f"  '{name}': {len(ids)} clusters")
    
    # Load embedding model
    print("\n[INFO] Step 2: Loading embedding model...")
    model = load_embedding_model()
    
    # Prepare texts for embedding: "cluster_name: definition"
    print("\n[INFO] Step 3: Preparing texts for embedding...")
    texts = []
    for _, row in df.iterrows():
        name = str(row['cluster_name']).strip()
        definition = str(row['definition']).strip()
        text = f"{name}: {definition}"
        texts.append(text)
    
    # Compute embeddings
    embeddings = compute_embeddings(model, texts)
    print(f"[INFO] Computed embeddings with shape {embeddings.shape}")
    
    # Find semantic matches
    print("\n[INFO] Step 4: Finding semantic matches...")
    semantic_matches = find_semantic_matches(df, embeddings, args.threshold)
    
    # Show sample matches
    if semantic_matches:
        print(f"\n[INFO] Sample semantic matches:")
        for id1, id2, sim in semantic_matches[:5]:
            name1 = df[df['cluster_id'] == id1]['cluster_name'].iloc[0]
            name2 = df[df['cluster_id'] == id2]['cluster_name'].iloc[0]
            print(f"  Cluster {id1} ('{name1}') <-> Cluster {id2} ('{name2}'): {sim:.3f}")
    
    # Create merge groups
    print("\n[INFO] Step 5: Creating merge groups...")
    cluster_to_merged = create_merge_groups(exact_matches, semantic_matches)
    
    # Assign merged names
    print("\n[INFO] Step 6: Assigning merged cluster IDs and names...")
    df_merged = assign_merged_names(df, cluster_to_merged, args.seed)
    
    # Save output
    print(f"\n[INFO] Saving output to {args.output}...")
    df_merged.to_csv(args.output, index=False)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Original clusters: {len(df)}")
    print(f"Exact name matches: {sum(len(ids) for ids in exact_matches.values())} clusters in {len(exact_matches)} groups")
    print(f"Semantic matches: {len(semantic_matches)} pairs")
    print(f"Unique merged clusters: {df_merged['cluster_merged_id'].nunique()}")
    print(f"Clusters merged: {len(df) - df_merged['cluster_merged_id'].nunique()}")
    print(f"Reduction: {(1 - df_merged['cluster_merged_id'].nunique() / len(df)) * 100:.1f}%")
    
    # Show merge examples
    print(f"\n[INFO] Merge examples:")
    merge_examples = df_merged[df_merged['cluster_id'] != df_merged['cluster_merged_id']].head(5)
    for _, row in merge_examples.iterrows():
        print(f"  Cluster {row['cluster_id']} ('{row['cluster_name']}') "
              f"-> Merged {row['cluster_merged_id']} ('{row['cluster_merged_name']}')")
    
    print(f"\n[INFO] Done!")
    print(f"[INFO] Output saved to: {args.output}")


if __name__ == "__main__":
    main()