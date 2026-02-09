#!/usr/bin/env python3
"""
K-means clustering with embeddings and centroids

Features:
- Saves embeddings for all instances
- Computes and saves cluster centroids
- Enables selecting instances closest to centroids for better cluster naming

Outputs:
- data/clusters.csv
- data/embeddings.npy (all instance embeddings)
- data/cluster_centroids.npy (centroid for each cluster)
"""

import argparse
import os
import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from transformers.utils import logging as hf_logging
from sklearn.cluster import KMeans
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


NUM_AUGMENTATIONS = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance = 1 - cosine_similarity"""
    a = nn.functional.normalize(a, p=2, dim=-1)
    b = nn.functional.normalize(b, p=2, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


class CompleteGroupsDataset(Dataset):
    """Dataset for complete groups only."""

    def __init__(self, df, aug_groups, tokenizer, max_length, pair_delim="|"):
        self.df = df
        self.aug_groups = aug_groups
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pair_delim = pair_delim
        
        self.inst_to_idx = {str(row['instance_id']): idx for idx, row in df.iterrows()}
        self.base_ids = list(aug_groups.keys())
        
        print(f"[INFO] Dataset: {len(self.base_ids)} complete groups")
        print(f"[INFO] Each group has exactly {NUM_AUGMENTATIONS} augmentations")
        
        self._validate()
    
    def _validate(self):
        """Quick validation that all groups are complete."""
        expected_size = 1 + NUM_AUGMENTATIONS
        
        for base_id, group in list(self.aug_groups.items())[:10]:
            if len(group) != expected_size:
                raise ValueError(
                    f"Group {base_id} has {len(group)} members, expected {expected_size}. "
                    f"Ensure input data contains only complete augmentation groups."
                )
        
        print(f"[INFO] ✓ Validation passed: All groups have {expected_size} members")
    
    def __len__(self):
        return len(self.base_ids)
    
    def _encode_instance(self, inst_id):
        """Encode a single instance."""
        row = self.df.loc[self.inst_to_idx[inst_id]]
        
        pred = str(row['predicate_lemma']).strip().lower()
        obj = str(row['object_lemma']).strip().lower()
        sent = str(row['sentence'])
        prefix = f"{pred}{self.pair_delim}{obj}"
        
        enc = self.tokenizer(
            prefix, sent,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).squeeze(0),
        }
    
    def __getitem__(self, idx):
        base_id = self.base_ids[idx]
        group = self.aug_groups[base_id]
        
        anchor_enc = self._encode_instance(base_id)
        aug_ids = [inst_id for inst_id in group if inst_id != base_id]
        
        assert len(aug_ids) == NUM_AUGMENTATIONS, \
            f"Group {base_id} has {len(aug_ids)} augmentations, expected {NUM_AUGMENTATIONS}"
        
        positives_enc = [self._encode_instance(aid) for aid in aug_ids]
        
        return {
            'a_input_ids': anchor_enc['input_ids'],
            'a_attention_mask': anchor_enc['attention_mask'],
            'a_token_type_ids': anchor_enc['token_type_ids'],
            'p_input_ids': torch.stack([p['input_ids'] for p in positives_enc]),
            'p_attention_mask': torch.stack([p['attention_mask'] for p in positives_enc]),
            'p_token_type_ids': torch.stack([p['token_type_ids'] for p in positives_enc]),
            'base_id': base_id,
        }


class BertCLSEncoder(nn.Module):
    """BERT encoder."""
    
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return out.last_hidden_state[:, 0, :]


def sample_negatives(anchors, base_ids, batch_size, aug_groups):
    """Sample negatives from different base_ids."""
    negatives = []
    stats = {'valid': 0, 'fallback': 0}
    
    for i in range(batch_size):
        anchor_base = base_ids[i]
        valid_idx = [j for j in range(batch_size) if base_ids[j] != anchor_base]
        
        if valid_idx:
            neg_idx = random.choice(valid_idx)
            stats['valid'] += 1
        else:
            neg_idx = (i + 1) % batch_size
            stats['fallback'] += 1
        
        negatives.append(anchors[neg_idx])
    
    return torch.stack(negatives), stats


@torch.no_grad()
def encode_all(encoder, tokenizer, df, max_length, device, pair_delim):
    """Encode all instances."""
    encoder.eval()
    embs = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        pred = str(row['predicate_lemma']).strip().lower()
        obj = str(row['object_lemma']).strip().lower()
        sent = str(row['sentence'])
        prefix = f"{pred}{pair_delim}{obj}"
        
        enc = tokenizer(
            prefix, sent,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        tti = enc.get("token_type_ids", torch.zeros_like(ids)).to(device)
        
        cls = encoder(ids, attn, tti)
        cls = nn.functional.normalize(cls, p=2, dim=-1)
        embs.append(cls.squeeze(0).cpu().numpy())
    
    return np.stack(embs)


def cluster_and_get_centroids(x, k, seed, use_faiss, gpu):
    """
    K-means clustering with centroid extraction.
    
    Returns:
        cluster_ids: Cluster assignment for each instance
        centroids: Centroid vectors for each cluster
    """
    if use_faiss and FAISS_AVAILABLE:
        d, k = x.shape[1], int(k)
        clus = faiss.Clustering(d, k)
        clus.verbose, clus.niter, clus.nredo, clus.seed = True, 20, 5, seed
        
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16, cfg.device = False, gpu
        idx = faiss.GpuIndexFlatL2(res, d, cfg)
        
        clus.train(x, idx)
        _, I = idx.search(x, 1)
        cluster_ids = np.array([int(n[0]) for n in I])
        
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
        
        return cluster_ids, centroids
    else:
        try:
            km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        except TypeError:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        
        cluster_ids = km.fit_predict(x)
        centroids = km.cluster_centers_
        
        return cluster_ids, centroids


def main():
    hf_logging.set_verbosity_error()
    
    ap = argparse.ArgumentParser(description="Train with centroid extraction")
    ap.add_argument("--input", default="data/train_aug_instances_complete.csv")
    ap.add_argument("--aug-groups", default="data/aug_groups.json")
    ap.add_argument("--output", default="data/clusters.csv")
    ap.add_argument("--output-embeddings", default="data/embeddings.npy")
    ap.add_argument("--output-centroids", default="data/cluster_centroids.npy")
    ap.add_argument("--model-name", default="bert-base-uncased")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--pair-delim", default="|")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.75)
    ap.add_argument("--num-cluster", type=int, default=50)
    ap.add_argument("--use-faiss", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu", type=int, default=0)
    
    args = ap.parse_args()
    set_seed(args.seed)
    
    print(f"[INFO] Loading complete groups from {args.aug_groups}...")
    with open(args.aug_groups) as f:
        aug_groups = json.load(f)
    print(f"[INFO] Loaded {len(aug_groups)} complete groups")
    
    print(f"[INFO] Loading complete instances from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} instances")
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    device = torch.device(args.device)
    encoder = BertCLSEncoder(args.model_name).to(device)
    
    ds = CompleteGroupsDataset(df, aug_groups, tokenizer, args.max_length, args.pair_delim)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    optim = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"\n{'='*80}")
    print("TRAINING ON COMPLETE GROUPS")
    print(f"{'='*80}")
    print(f"  Groups: {len(aug_groups)}")
    print(f"  Positives per instance: {NUM_AUGMENTATIONS} (fixed)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    encoder.train()
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        steps = 0
        stats = defaultdict(int)
        
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            a_ids = batch["a_input_ids"].to(device)
            a_attn = batch["a_attention_mask"].to(device)
            a_tti = batch["a_token_type_ids"].to(device)
            
            p_ids = batch["p_input_ids"].to(device)
            p_attn = batch["p_attention_mask"].to(device)
            p_tti = batch["p_token_type_ids"].to(device)
            
            base_ids = batch["base_id"]
            
            a = encoder(a_ids, a_attn, a_tti)
            
            B, N, L = p_ids.shape
            p_ids_flat = p_ids.view(B * N, L)
            p_attn_flat = p_attn.view(B * N, L)
            p_tti_flat = p_tti.view(B * N, L)
            
            p_flat = encoder(p_ids_flat, p_attn_flat, p_tti_flat)
            p = p_flat.view(B, N, -1)
            
            n, batch_stats = sample_negatives(a.detach(), base_ids, args.batch_size, aug_groups)
            
            for k, v in batch_stats.items():
                stats[k] += v
            
            loss = 0.0
            for i in range(NUM_AUGMENTATIONS):
                p_i = p[:, i, :]
                d_ap = cosine_distance(a, p_i)
                d_an = cosine_distance(a, n)
                loss += torch.relu(d_ap - d_an + args.margin).mean()
            
            loss = loss / NUM_AUGMENTATIONS
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / steps
        print(f"\n[INFO] Epoch {epoch+1}: loss={avg_loss:.4f}, valid_neg={stats['valid']}, fallback={stats['fallback']}")
    
    print(f"\n[INFO] Training complete!")
    
    print(f"\n[INFO] Encoding all instances...")
    embeddings = encode_all(encoder, tokenizer, df, args.max_length, device, args.pair_delim)
    
    print(f"\n[INFO] Clustering into {args.num_cluster} clusters...")
    cluster_ids, centroids = cluster_and_get_centroids(
        embeddings, args.num_cluster, args.seed, args.use_faiss, args.gpu
    )
    
    df['cluster_id'] = cluster_ids
    df.to_csv(args.output, index=False)
    print(f"[INFO] ✓ Saved clusters to {args.output}")
    
    np.save(args.output_embeddings, embeddings)
    print(f"[INFO] ✓ Saved embeddings to {args.output_embeddings}")
    
    np.save(args.output_centroids, centroids)
    print(f"[INFO] ✓ Saved centroids to {args.output_centroids}")
    
    print(f"\n[INFO] Cluster distribution:")
    for cid, count in Counter(cluster_ids).most_common(10):
        print(f"  Cluster {cid}: {count}")
    
    print(f"\n✓ Done!")
    print(f"\nFiles created:")
    print(f"  - {args.output} (clusters with IDs)")
    print(f"  - {args.output_embeddings} (instance embeddings)")
    print(f"  - {args.output_centroids} (cluster centroids)")


if __name__ == "__main__":
    main()
