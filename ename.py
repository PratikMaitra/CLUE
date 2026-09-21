#!/usr/bin/env python3
"""
6_event_naming_no_leakage.py - Name k-means clusters using LLM in-context learning

NO-LEAKAGE VERSION: Ground truth (gold) labels are NOT shown to the LLM.
The LLM must infer the event type purely from the unlabeled examples
(sentence + predicate + object). All references to the ACE dataset have
also been removed from the prompt to avoid priming the model toward a
known ontology.

Uses an LLM via OpenRouter API to name clusters based on 5 example instances
WITHOUT any ground truth event labels.

Input:
  - data/clusters.csv

Outputs:
  - data/clusters_named.csv: Original file + cluster_name column
  - data/cluster_names.csv: Mapping of cluster_id -> cluster_name
  - data/prompts.csv: Full prompts and LLM responses for each cluster

Process:
  1. Read clusters.csv
  2. For each cluster, select 5 example instances closest to the centroid
  3. Build prompt with UNLABELED examples (no gold labels, no ACE mention)
  4. Send to LLM via OpenRouter
  5. Save cluster names and prompts
"""

import os
import json
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


# ---------------- DEFAULT SETTINGS ----------------
# SECURITY: Do NOT hardcode API keys in source code.
# Set the key via:  export OPENROUTER_API_KEY="sk-or-..."
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4-turbo"  # or "openai/gpt-4o" or other models

TOP_K_EXAMPLES = 5


def build_prompt(cluster_id: int, examples: List[Dict]) -> str:
    """
    Build a prompt for naming one cluster using in-context learning.

    NO-LEAKAGE: Examples contain ONLY sentence, predicate, and object.
    No ground truth labels are included, and no dataset/ontology is named.

    Args:
        cluster_id: The cluster ID being named
        examples: List of dicts with keys: predicate_lemma, object_lemma, sentence

    Returns:
        Formatted prompt string
    """
    header = (
        "You are an expert on event extraction.\n"
        "Your task is to infer the EVENT TYPE for a cluster of event instances.\n\n"
        "Details:\n"
        "- Below are several examples from the same cluster.\n"
        "- Each example shows: a sentence and its predicate-object pair.\n"
        "- All examples belong to the same cluster and should share a common event type.\n\n"
        "Your job:\n"
        "1. Read the examples and identify the event they have in common.\n"
        "2. Respond with ONLY a concise event type name in the format \"Category:Subtype\".\n"
        "3. Output the event type name in <> tags.\n"
        "4. Do NOT output explanations, reasoning, or anything else - only the event type.\n\n"
        f"Here are {len(examples)} examples from Cluster {cluster_id}:\n\n"
    )

    body_lines = []
    for i, ex in enumerate(examples, start=1):
        pred_lemma = ex.get('predicate_lemma', '')
        obj_lemma = ex.get('object_lemma', '')
        sentence = ex.get('sentence', '')

        body_lines.append(
            f"Example {i}:\n"
            f"  Sentence: {sentence}\n"
            f"  Predicate: {pred_lemma}\n"
            f"  Object: {obj_lemma}\n"
        )

    prompt = header + "\n".join(body_lines)
    prompt += (
        "\n\nBased on these examples, what is the event type for this cluster?\n"
        "Output format: <Category:Subtype>\n"
    )

    return prompt


def call_openrouter(prompt: str, api_key: str, model: str) -> str:
    """
    Call OpenRouter API with the given prompt.

    Args:
        prompt: The prompt to send
        api_key: OpenRouter API key
        model: Model identifier (e.g., "openai/gpt-4-turbo")

    Returns:
        LLM response text (extracted from <> tags if present)
    """
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is not set. "
            "Provide it via --api-key argument or set OPENROUTER_API_KEY environment variable."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful and concise assistant that names event types based on unlabeled examples.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3,
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Extract response text
        text = data["choices"][0]["message"]["content"].strip()

        # Extract content from <> tags if present
        import re
        tag_match = re.search(r'<([^>]+)>', text)
        if tag_match:
            text = tag_match.group(1).strip()
        else:
            # Keep only first line (in case LLM outputs multiple lines)
            text = text.splitlines()[0].strip()

        return text

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] OpenRouter API call failed: {e}")
        return "ERROR"


def select_examples(cluster_df: pd.DataFrame, embeddings: np.ndarray, centroids: np.ndarray,
                   cluster_id: int, k: int = 5) -> List[Dict]:
    """
    Select k example instances closest to the cluster centroid.

    This selects the most representative instances from the cluster.

    NO-LEAKAGE: The returned example dicts do NOT contain any gold labels.

    Args:
        cluster_df: DataFrame containing all instances in the cluster
        embeddings: All instance embeddings (n_instances, embedding_dim)
        centroids: All cluster centroids (n_clusters, embedding_dim)
        cluster_id: The cluster ID
        k: Number of examples to select

    Returns:
        List of example dicts (sentence, predicate, object only)
    """
    if len(cluster_df) == 0:
        return []

    # Get centroid for this cluster
    centroid = centroids[cluster_id]

    # Get embeddings for instances in this cluster (by index)
    instance_indices = cluster_df.index.tolist()
    cluster_embeddings = embeddings[instance_indices]

    # Compute cosine distances to centroid
    from scipy.spatial.distance import cdist
    centroid_reshaped = centroid.reshape(1, -1)
    distances = cdist(cluster_embeddings, centroid_reshaped, metric='cosine').flatten()

    # Add distance to dataframe
    cluster_df_with_dist = cluster_df.copy()
    cluster_df_with_dist['distance_to_centroid'] = distances

    # Prefer original instances (without _aug suffix)
    if 'is_augmented' in cluster_df_with_dist.columns:
        original_mask = ~cluster_df_with_dist['is_augmented']
        original_df = cluster_df_with_dist[original_mask].sort_values('distance_to_centroid')
        augmented_df = cluster_df_with_dist[~original_mask].sort_values('distance_to_centroid')

        if len(original_df) >= k:
            selected = original_df.head(k)
        else:
            need_more = k - len(original_df)
            selected = pd.concat([original_df, augmented_df.head(need_more)])
    else:
        # No is_augmented column, just sort by distance
        selected = cluster_df_with_dist.sort_values('distance_to_centroid').head(k)

    # Convert to list of dicts -- NO gold labels included
    examples = []
    for _, row in selected.iterrows():
        examples.append({
            'predicate_lemma': row.get('predicate_lemma', ''),
            'object_lemma': row.get('object_lemma', ''),
            'sentence': row.get('sentence', ''),
            'instance_id': row.get('instance_id', ''),
            'distance_to_centroid': row.get('distance_to_centroid', 0.0),
        })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Name k-means clusters using LLM in-context learning (WITHOUT ground truth labels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        '--input',
        default='data/clusters.csv',
        help='Input clusters file (default: data/clusters.csv)'
    )
    parser.add_argument(
        '--embeddings',
        default='data/embeddings.npy',
        help='Input embeddings file (default: data/embeddings.npy)'
    )
    parser.add_argument(
        '--centroids',
        default='data/cluster_centroids.npy',
        help='Input centroids file (default: data/cluster_centroids.npy)'
    )
    parser.add_argument(
        '--output-clusters',
        default='data/clusters_named.csv',
        help='Output file with cluster names (default: data/clusters_named.csv)'
    )
    parser.add_argument(
        '--output-names',
        default='data/cluster_names.csv',
        help='Output file with cluster ID to name mapping (default: data/cluster_names.csv)'
    )
    parser.add_argument(
        '--output-prompts',
        default='data/prompts.csv',
        help='Output file with prompts and responses (default: data/prompts.csv)'
    )

    # LLM settings
    parser.add_argument(
        '--api-key',
        default=OPENROUTER_API_KEY,
        help='OpenRouter API key (default: from OPENROUTER_API_KEY environment variable)'
    )
    parser.add_argument(
        '--model',
        default=OPENROUTER_MODEL,
        help=f'LLM model to use (default: {OPENROUTER_MODEL})'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=TOP_K_EXAMPLES,
        help=f'Number of examples per cluster (default: {TOP_K_EXAMPLES})'
    )

    # System settings
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate prompts without calling LLM'
    )

    args = parser.parse_args()

    # Check API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key and not args.dry_run:
        print("[ERROR] No OpenRouter API key provided.")
        print("Provide via --api-key argument or OPENROUTER_API_KEY environment variable.")
        print("Or use --dry-run to generate prompts only.")
        return

    # Load clusters
    print(f"[INFO] Reading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} instances")

    # NOTE: gold labels (e.g. 'gold_event_types') may exist in the input file
    # for later EVALUATION purposes, but they are never passed to the LLM.

    # Load embeddings and centroids
    print(f"\n[INFO] Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"[INFO] Loaded embeddings: shape {embeddings.shape}")

    print(f"[INFO] Loading centroids from {args.centroids}...")
    centroids = np.load(args.centroids)
    print(f"[INFO] Loaded centroids: shape {centroids.shape}")

    # Get unique cluster IDs
    cluster_ids = sorted(df['cluster_id'].unique())
    print(f"[INFO] Found {len(cluster_ids)} unique clusters")

    # Process each cluster
    cluster_names = {}
    prompt_records = []

    for cluster_id in tqdm(cluster_ids, desc="Naming clusters"):
        # Get all instances in this cluster
        cluster_df = df[df['cluster_id'] == cluster_id]

        # Select k examples closest to centroid
        examples = select_examples(cluster_df, embeddings, centroids, cluster_id, k=args.num_examples)

        # Build prompt (unlabeled examples only)
        prompt = build_prompt(cluster_id, examples)

        # Call LLM (or skip if dry-run)
        if args.dry_run:
            cluster_name = f"CLUSTER_{cluster_id}"
            print(f"\n[DRY RUN] Cluster {cluster_id} prompt:\n{prompt}\n")
        else:
            try:
                cluster_name = call_openrouter(prompt, api_key, args.model)
                print(f"[INFO] Cluster {cluster_id}: {cluster_name}")
            except Exception as e:
                print(f"[ERROR] Failed to name cluster {cluster_id}: {e}")
                cluster_name = f"ERROR_CLUSTER_{cluster_id}"

        # Store cluster name
        cluster_names[cluster_id] = cluster_name

        # Record prompt and response
        prompt_records.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'LLM_model': args.model,
            'prompt': prompt,
            'LLM_response': cluster_name,
        })

    # Add cluster_name column to original dataframe
    print(f"\n[INFO] Adding cluster names to dataframe...")
    df['cluster_name'] = df['cluster_id'].map(cluster_names)

    # Save outputs
    print(f"[INFO] Saving outputs...")

    # 1. Save clusters_named.csv (original + cluster_name column)
    df.to_csv(args.output_clusters, index=False)
    print(f"[INFO] Saved {args.output_clusters}")

    # 2. Save cluster_names.csv (cluster_id -> cluster_name mapping)
    cluster_names_df = pd.DataFrame([
        {'cluster_id': cid, 'cluster_name': cname}
        for cid, cname in cluster_names.items()
    ])
    cluster_names_df.to_csv(args.output_names, index=False)
    print(f"[INFO] Saved {args.output_names}")

    # 3. Save prompts.csv (full prompts and responses)
    prompts_df = pd.DataFrame(prompt_records)
    prompts_df.to_csv(args.output_prompts, index=False)
    print(f"[INFO] Saved {args.output_prompts}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total clusters: {len(cluster_ids)}")
    print(f"Successfully named: {len([n for n in cluster_names.values() if not n.startswith('ERROR')])}")
    print(f"Errors: {len([n for n in cluster_names.values() if n.startswith('ERROR')])}")
    print(f"\nSample cluster names:")
    for cid in list(cluster_ids)[:5]:
        print(f"  Cluster {cid}: {cluster_names[cid]}")
    print(f"\n[INFO] Done!")


if __name__ == "__main__":
    main()
