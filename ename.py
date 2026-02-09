#!/usr/bin/env python3


import os
import json
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


# ---------------- PLEASE FILL IN WITH API KEY  ----------------
OPENROUTER_API_KEY = ""
OPENROUTER_URL = ""
OPENROUTER_MODEL = "openai/gpt-4-turbo"  

TOP_K_EXAMPLES = 5


def build_prompt(cluster_id: int, examples: List[Dict]) -> str:
    
    header = (
        "You are an expert on ACE-style event extraction.\n"
        "Your task is to infer the EVENT TYPE for a cluster based on labeled examples.\n\n"
        "Details:\n"
        "- Below are several examples from the same cluster.\n"
        "- Each example shows: sentence, predicate-object pair, and GROUND TRUTH event label.\n"
        "- All examples belong to the same cluster and should share a common event type.\n"
        "- The dataset follows ACE (Automatic Content Extraction) event ontology.\n\n"
        "Your job:\n"
        "1. Look at the ground truth labels of the examples.\n"
        "2. Respond with ONLY an appropriate event type name, keeping in mind ACE event ontology.\n"
        "   Example outputs: \"Conflict:Attack\", \"Movement:Transport\", \"Life:Die\", \"Contact:Meet\", \"Transaction:Transfer-Money\", \"Justice:Execute\", \"Personnel:Nominate\", \"Business:Merge-Org\".\n"
        "3. Output the event type name in <> tags.\n"
        "4. Do NOT output explanations, reasoning, or anything else - only the event type.\n\n"
        f"Here are {len(examples)} labeled examples from Cluster {cluster_id}:\n\n"
    )

    body_lines = []
    for i, ex in enumerate(examples, start=1):
        pred_lemma = ex.get('predicate_lemma', '')
        obj_lemma = ex.get('object_lemma', '')
        sentence = ex.get('sentence', '')
        event_type = ex.get('event_type', 'UNKNOWN')  # ← KEY ADDITION
        
        body_lines.append(
            f"Example {i}:\n"
            f"  Sentence: {sentence}\n"
            f"  Predicate: {pred_lemma}\n"
            f"  Object: {obj_lemma}\n"
            f"  Ground Truth Label: {event_type}\n"  # ← SHOW THE LABEL
        )

    prompt = header + "\n".join(body_lines)
    prompt += (
        "\n\nBased on these labeled examples, what is the event type for this cluster?\n"
        "Output format: <EventType>\n"
    )
    
    return prompt


def call_openrouter(prompt: str, api_key: str, model: str) -> str:
    
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
                "content": "You are a careful and concise assistant that names ACE-style event types based on labeled examples.",
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
        
        
        text = data["choices"][0]["message"]["content"].strip()
        
        
        import re
        tag_match = re.search(r'<([^>]+)>', text)
        if tag_match:
            text = tag_match.group(1).strip()
        else:
            
            text = text.splitlines()[0].strip()
        
        return text
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] OpenRouter API call failed: {e}")
        return "ERROR"


def select_examples(cluster_df: pd.DataFrame, embeddings: np.ndarray, centroids: np.ndarray, 
                   cluster_id: int, k: int = 5) -> List[Dict]:
    
    if len(cluster_df) == 0:
        return []
    
    
    centroid = centroids[cluster_id]
    
    
    instance_indices = cluster_df.index.tolist()
    cluster_embeddings = embeddings[instance_indices]
    
    
    from scipy.spatial.distance import cdist
    centroid_reshaped = centroid.reshape(1, -1)
    distances = cdist(cluster_embeddings, centroid_reshaped, metric='cosine').flatten()
    
    
    cluster_df_with_dist = cluster_df.copy()
    cluster_df_with_dist['distance_to_centroid'] = distances
    
    
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
        
        selected = cluster_df_with_dist.sort_values('distance_to_centroid').head(k)
    
    
    examples = []
    for _, row in selected.iterrows():
        examples.append({
            'predicate_lemma': row.get('predicate_lemma', ''),
            'object_lemma': row.get('object_lemma', ''),
            'sentence': row.get('sentence', ''),
            'instance_id': row.get('instance_id', ''),
            'event_type': row.get('event_type', 'UNKNOWN'),
            'distance_to_centroid': row.get('distance_to_centroid', 0.0),
        })
    
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Name k-means clusters using LLM in-context learning (with ground truth labels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    
    parser.add_argument(
        '--input',
        default='data/clusters.csv',
        help='Input clusters file (default: data/clusters.csv - must have gold_event_types column)'
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
    
    
    parser.add_argument(
        '--api-key',
        default=OPENROUTER_API_KEY,
        help='OpenRouter API key (default: from script or env)'
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
    
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate prompts without calling LLM'
    )
    
    args = parser.parse_args()
    
    
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key and not args.dry_run:
        print("[ERROR] No OpenRouter API key provided.")
        print("Provide via --api-key argument or OPENROUTER_API_KEY environment variable.")
        print("Or use --dry-run to generate prompts only.")
        return
    
   
    print(f"[INFO] Reading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} instances")
    
    
    if 'gold_event_types' not in df.columns:
        print("[ERROR] Input file must have 'gold_event_types' column!")
        print("This column should have been added in step 1 (extract_predicate_object_with_gold.py)")
        return
    
    
    df['event_type'] = df['gold_event_types']
    print(f"[INFO] Using 'gold_event_types' column for ground truth labels")
    
    
    print(f"\n[INFO] Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"[INFO] Loaded embeddings: shape {embeddings.shape}")
    
    print(f"[INFO] Loading centroids from {args.centroids}...")
    centroids = np.load(args.centroids)
    print(f"[INFO] Loaded centroids: shape {centroids.shape}")
    
    
    cluster_ids = sorted(df['cluster_id'].unique())
    print(f"[INFO] Found {len(cluster_ids)} unique clusters")
    
    
    cluster_names = {}
    prompt_records = []
    
    for cluster_id in tqdm(cluster_ids, desc="Naming clusters"):
        
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        
        examples = select_examples(cluster_df, embeddings, centroids, cluster_id, k=args.num_examples)
        
        
        prompt = build_prompt(cluster_id, examples)
        
        
        if args.dry_run:
            cluster_name = f"CLUSTER_{cluster_id}"
            print(f"\n[DRY RUN] Cluster {cluster_id} prompt:\n{prompt}\n")
        else:
            try:
                cluster_name = call_openrouter(prompt, api_key, args.model)
                
                
                gt_labels = cluster_df['event_type'].value_counts()
                print(f"[INFO] Cluster {cluster_id}: {cluster_name}")
                print(f"       Ground truth distribution: {dict(gt_labels)}")
                
            except Exception as e:
                print(f"[ERROR] Failed to name cluster {cluster_id}: {e}")
                cluster_name = f"ERROR_CLUSTER_{cluster_id}"
        
        
        cluster_names[cluster_id] = cluster_name
        
       
        prompt_records.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'LLM_model': args.model,
            'prompt': prompt,
            'LLM_response': cluster_name,
        })
    
    
    print(f"\n[INFO] Adding cluster names to dataframe...")
    df['cluster_name'] = df['cluster_id'].map(cluster_names)
    
    
    print(f"[INFO] Saving outputs...")
    
    
    df.to_csv(args.output_clusters, index=False)
    print(f"[INFO] Saved {args.output_clusters}")
    
    
    cluster_names_df = pd.DataFrame([
        {'cluster_id': cid, 'cluster_name': cname}
        for cid, cname in cluster_names.items()
    ])
    cluster_names_df.to_csv(args.output_names, index=False)
    print(f"[INFO] Saved {args.output_names}")
    
    
    prompts_df = pd.DataFrame(prompt_records)
    prompts_df.to_csv(args.output_prompts, index=False)
    print(f"[INFO] Saved {args.output_prompts}")
    
   
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