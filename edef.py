#!/usr/bin/env python3


import os
import json
import argparse
from typing import Dict
import pandas as pd
import requests
from tqdm import tqdm


# ---------------- PLEASE FILL IN WITH API KEY ----------------
OPENROUTER_API_KEY = ""
OPENROUTER_URL = ""
OPENROUTER_MODEL = "openai/gpt-4-turbo"


def build_definition_prompt(cluster_name: str) -> str:
    
    prompt = (
        "You are an expert on ACE-style event extraction.\n\n"
        f"Your task: Write a fine-grained definition of the event type \"{cluster_name}\" in 2-4 sentences.\n\n"
        "Details:\n"
        "- The definition should clearly explain what this event type represents.\n"
        "- It should be specific enough to distinguish it from related event types.\n"
        "- Focus on the key characteristics, participants, and typical actions involved.\n"
        "- Use clear, concise language suitable for event extraction guidelines.\n\n"
        "Requirements:\n"
        "- Write exactly 2-4 sentences.\n"
        "- Be precise and informative.\n"
        "- Do NOT include examples or list formats.\n"
        "- Output ONLY the definition text, nothing else.\n\n"
        f"Event type: {cluster_name}\n\n"
        "Definition:"
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
                "content": "You are an expert on ACE-style event extraction who writes clear, precise event type definitions.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3,
        "max_tokens": 200,  
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        
        text = data["choices"][0]["message"]["content"].strip()
        
        return text
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] OpenRouter API call failed: {e}")
        return "ERROR: Failed to generate definition"


def main():
    parser = argparse.ArgumentParser(
        description="Generate fine-grained definitions for cluster names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    
    parser.add_argument(
        '--input',
        default='data/cluster_names.csv',
        help='Input cluster names file (default: data/cluster_names.csv)'
    )
    parser.add_argument(
        '--output-definitions',
        default='data/cluster_names_definition.csv',
        help='Output file with definitions (default: data/cluster_names_definition.csv)'
    )
    parser.add_argument(
        '--output-prompts',
        default='data/prompts_definition.csv',
        help='Output file with prompts and responses (default: data/prompts_definition.csv)'
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
    print(f"[INFO] Loaded {len(df)} cluster names")
    
    
    definitions = []
    prompt_records = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating definitions"):
        cluster_id = row['cluster_id']
        cluster_name = row['cluster_name']
        
        
        prompt = build_definition_prompt(cluster_name)
        
        
        if args.dry_run:
            definition = f"[DRY RUN] Definition for {cluster_name}"
            print(f"\n[DRY RUN] Cluster {cluster_id} ({cluster_name}) prompt:\n{prompt}\n")
        else:
            try:
                definition = call_openrouter(prompt, api_key, args.model)
                print(f"[INFO] Cluster {cluster_id} ({cluster_name}): {definition[:80]}...")
            except Exception as e:
                print(f"[ERROR] Failed to get definition for cluster {cluster_id}: {e}")
                definition = f"ERROR: Failed to generate definition"
        
        
        definitions.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'definition': definition,
        })
        
        
        prompt_records.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'LLM_model': args.model,
            'prompt': prompt,
            'LLM_response': definition,
        })
    
    
    print(f"\n[INFO] Saving outputs...")
    
    
    definitions_df = pd.DataFrame(definitions)
    definitions_df.to_csv(args.output_definitions, index=False)
    print(f"[INFO] Saved {args.output_definitions}")
    
    
    prompts_df = pd.DataFrame(prompt_records)
    prompts_df.to_csv(args.output_prompts, index=False)
    print(f"[INFO] Saved {args.output_prompts}")
    
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total cluster names: {len(df)}")
    print(f"Successfully defined: {len([d for d in definitions if not d['definition'].startswith('ERROR')])}")
    print(f"Errors: {len([d for d in definitions if d['definition'].startswith('ERROR')])}")
    
    
    print(f"\nSample definitions:")
    for i, defn in enumerate(definitions[:3]):
        print(f"\n{i+1}. {defn['cluster_name']}:")
        print(f"   {defn['definition']}")
    
    print(f"\n[INFO] Done!")


if __name__ == "__main__":
    main()