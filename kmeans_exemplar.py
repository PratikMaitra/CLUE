#!/usr/bin/env python3
"""
5b_cluster_contrastive_exemplar.py - Contrastive-loss-guided K-Means exemplar clustering

Implements the loss of AugURE (Wang et al., EMNLP 2023, Eq. 8-10), which CLUE
follows for its clustering stage:

    L = L_Pair + L_Exem

  L_Pair (Eq. 8)  margin triplet loss, cosine distance, in-batch random negative
                  anchor = original instance, positives = its MLM augmentations
                      L_Pair = mean max{ d(a, p) - d(a, n) + gamma, 0 }

  L_Exem (Eq. 9)  exemplar loss of Liu et al. (2022): softmax over exemplars
                      L_Exem = - mean_i (1/L) sum_l log softmax_j( h_i . e^l_j / tau )
                  where e^l_j are the K-Means CENTROIDS at granularity l
                  (AugURE 4.3.1: "cluster centroids of different k values as
                  different granularities of relational exemplars"), recomputed
                  each epoch as the encoder updates.

Schedule
  epoch 1 .. warmup     : L_Pair only (no clusters yet)
  epoch warmup+1 .. E   : at epoch start, encode corpus -> K-Means at each k in
                          --exemplar-ks -> exemplar bank; train with L_Pair + L_Exem
  after last epoch      : encode -> K-Means (k = --num-cluster) -> save

Outputs
  data/clusters.csv             cluster_id, exemplar_instance_id, dist_to_exemplar, is_exemplar
  data/embeddings.npy           all instance embeddings (n x d)
  data/cluster_centroids.npy    K-Means centroids = exemplars (k x d)
  data/cluster_exemplars.csv    per-cluster nearest real instance to the centroid
                                (for LLM naming / in-context examples)
"""

import argparse
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


NUM_AUGMENTATIONS = 5  # Fixed: all groups have exactly 5 augmentations


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(a, p=2, dim=-1)
    b = nn.functional.normalize(b, p=2, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_prefix_and_sentence(row, pair_delim):
    pred = str(row['predicate_lemma']).strip().lower()
    obj = str(row['object_lemma']).strip().lower()
    return f"{pred}{pair_delim}{obj}", str(row['sentence'])


class CompleteGroupsDataset(Dataset):
    """Complete groups only; also returns df row indices for the exemplar loss."""

    def __init__(self, df, aug_groups, tokenizer, max_length, pair_delim="|"):
        self.df = df
        self.aug_groups = aug_groups
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pair_delim = pair_delim
        self.inst_to_idx = {str(row['instance_id']): idx for idx, row in df.iterrows()}
        self.base_ids = list(aug_groups.keys())
        print(f"[INFO] Dataset: {len(self.base_ids)} complete groups")
        self._validate()

    def _validate(self):
        expected = 1 + NUM_AUGMENTATIONS
        for base_id, group in list(self.aug_groups.items())[:10]:
            if len(group) != expected:
                raise ValueError(
                    f"Group {base_id} has {len(group)} members, expected {expected}. "
                    f"Did you run 5a_prepare_augmentation_groups_COMPLETE_ONLY.py first?")
        print(f"[INFO] ✅ Validation passed: All groups have {expected} members")

    def __len__(self):
        return len(self.base_ids)

    def _encode_instance(self, inst_id):
        row_idx = self.inst_to_idx[inst_id]
        prefix, sent = build_prefix_and_sentence(self.df.loc[row_idx], self.pair_delim)
        enc = self.tokenizer(prefix, sent, add_special_tokens=True, max_length=self.max_length,
                             truncation=True, padding="max_length", return_tensors="pt")
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).squeeze(0),
            'row_idx': row_idx,
        }

    def __getitem__(self, idx):
        base_id = self.base_ids[idx]
        aug_ids = [i for i in self.aug_groups[base_id] if i != base_id]
        assert len(aug_ids) == NUM_AUGMENTATIONS
        a = self._encode_instance(base_id)
        ps = [self._encode_instance(i) for i in aug_ids]
        return {
            'a_input_ids': a['input_ids'], 'a_attention_mask': a['attention_mask'],
            'a_token_type_ids': a['token_type_ids'],
            'a_row_idx': torch.tensor(a['row_idx'], dtype=torch.long),
            'p_input_ids': torch.stack([p['input_ids'] for p in ps]),
            'p_attention_mask': torch.stack([p['attention_mask'] for p in ps]),
            'p_token_type_ids': torch.stack([p['token_type_ids'] for p in ps]),
            'p_row_idx': torch.tensor([p['row_idx'] for p in ps], dtype=torch.long),
            'base_id': base_id,
        }


class BertCLSEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        return out.last_hidden_state[:, 0, :]


def sample_negatives(anchors, base_ids, batch_size):
    """Random in-batch negative from a different base instance (AugURE Eq. 8, n_r)."""
    negatives, stats = [], {'valid': 0, 'fallback': 0}
    for i in range(batch_size):
        valid = [j for j in range(batch_size) if base_ids[j] != base_ids[i]]
        if valid:
            negatives.append(anchors[random.choice(valid)]); stats['valid'] += 1
        else:
            negatives.append(anchors[(i + 1) % batch_size]); stats['fallback'] += 1
    return torch.stack(negatives), stats


@torch.no_grad()
def encode_all(encoder, tokenizer, df, max_length, device, pair_delim, batch_size=256):
    """Batched encoding of every row. Returns L2-normalised float32 (n, d)."""
    encoder.eval()
    prefixes, sents = zip(*[build_prefix_and_sentence(r, pair_delim) for _, r in df.iterrows()])
    embs = []
    for s in tqdm(range(0, len(df), batch_size), desc="Encoding"):
        enc = tokenizer(list(prefixes[s:s + batch_size]), list(sents[s:s + batch_size]),
                        add_special_tokens=True, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        tti = enc.get("token_type_ids", torch.zeros_like(ids)).to(device)
        embs.append(nn.functional.normalize(encoder(ids, attn, tti), p=2, dim=-1).cpu().numpy())
    encoder.train()
    return np.concatenate(embs).astype(np.float32)


# ---------------------------------------------------------------------------
# K-Means exemplars (= centroids), one bank per granularity k
# ---------------------------------------------------------------------------

def run_kmeans(x, k, seed, use_faiss, gpu):
    """Returns (cluster_ids (n,), centroids (k,d))."""
    if use_faiss and FAISS_AVAILABLE:
        d = x.shape[1]
        clus = faiss.Clustering(d, int(k))
        clus.verbose, clus.niter, clus.nredo, clus.seed = False, 20, 5, seed
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16, cfg.device = False, gpu
        idx = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(x, idx)
        _, I = idx.search(x, 1)
        return np.array([int(n[0]) for n in I]), faiss.vector_to_array(clus.centroids).reshape(k, d)
    try:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    ids = km.fit_predict(x)
    return ids, km.cluster_centers_.astype(np.float32)


def medoid_exemplars(x, cluster_ids, k):
    """Optional variant: exemplar = real member minimising total cosine distance."""
    ex_idx = np.zeros(k, dtype=np.int64)
    for c in range(k):
        m = np.where(cluster_ids == c)[0]
        if len(m) == 0:
            ex_idx[c] = 0
        elif len(m) == 1:
            ex_idx[c] = m[0]
        else:
            ex_idx[c] = m[int(np.argmin((1.0 - x[m] @ x[m].T).sum(axis=1)))]
    return ex_idx


def build_exemplar_bank(encoder, tokenizer, df, args, device, seed):
    """
    Encode corpus, run K-Means at every k in args.exemplar_ks, and return one
    exemplar bank per granularity (AugURE 4.3.1 / Eq. 9 with L = len(ks)).
    Bank l = { 'E': (k_l, d) exemplar vectors, 'cluster_ids_t': (n,) }.
    """
    x = encode_all(encoder, tokenizer, df, args.max_length, device, args.pair_delim)
    banks = []
    for l, k in enumerate(args.exemplar_ks):
        ids, centroids = run_kmeans(x, k, seed + l, args.use_faiss, args.gpu)
        if args.exemplar_type == "centroid":
            E = centroids
        else:  # medoid: real instance per cluster
            E = x[medoid_exemplars(x, ids, k)]
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        banks.append({
            'k': k,
            'E': torch.from_numpy(E.astype(np.float32)).to(device),
            'cluster_ids_t': torch.from_numpy(ids).long().to(device),
            'cluster_ids': ids,
            'centroids': centroids,
        })
        sizes = np.bincount(ids, minlength=k)
        print(f"[INFO]   granularity l={l+1}: k={k}, cluster sizes min/max = {sizes.min()}/{sizes.max()}")
    return x, banks


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def pair_margin_loss(a, p, n, margin):
    """AugURE Eq. 8 / CLUE Eq. 5, averaged over the NUM_AUGMENTATIONS positives."""
    loss = 0.0
    for i in range(NUM_AUGMENTATIONS):
        loss = loss + torch.relu(cosine_distance(a, p[:, i, :]) - cosine_distance(a, n) + margin).mean()
    return loss / NUM_AUGMENTATIONS


def exemplar_loss(z, row_idx, banks, tau):
    """
    AugURE Eq. 9 (Liu et al. 2022): for each granularity l, softmax over all
    exemplars of the cosine similarity, averaged over l.
    """
    z = nn.functional.normalize(z, p=2, dim=-1)
    loss = 0.0
    for b in banks:
        logits = (z @ b['E'].T) / tau                     # (m, k_l)
        loss = loss + nn.functional.cross_entropy(logits, b['cluster_ids_t'][row_idx])
    return loss / len(banks)


# ---------------------------------------------------------------------------

def main():
    hf_logging.set_verbosity_error()
    ap = argparse.ArgumentParser(description="Contrastive (margin pair + exemplar) K-Means clustering")
    ap.add_argument("--input", default="data/train_aug_instances_complete.csv")
    ap.add_argument("--aug-groups", default="data/aug_groups.json")
    ap.add_argument("--output", default="data/clusters.csv")
    ap.add_argument("--output-embeddings", default="data/embeddings.npy")
    ap.add_argument("--output-centroids", default="data/cluster_centroids.npy")
    ap.add_argument("--output-exemplars", default="data/cluster_exemplars.csv")
    ap.add_argument("--model-name", default="bert-base-uncased")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--pair-delim", default="|")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--warmup-epochs", type=int, default=1,
                    help="epochs with L_Pair only before exemplars exist")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.75, help="gamma in L_Pair")
    ap.add_argument("--tau", type=float, default=0.02, help="temperature in L_Exem (AugURE App. A)")
    ap.add_argument("--exemplar-weight", type=float, default=1.0,
                    help="weight on L_Exem; AugURE Eq. 10 uses 1.0")
    ap.add_argument("--num-cluster", type=int, default=50,
                    help="k for the final clustering (and first exemplar granularity)")
    ap.add_argument("--exemplar-ks", default="",
                    help="comma-separated extra k values for multi-granularity exemplars, "
                         "e.g. '25,100'. Empty = single granularity at --num-cluster (L=1)")
    ap.add_argument("--exemplar-type", choices=["centroid", "medoid"], default="centroid",
                    help="centroid = K-Means centroid vectors (AugURE); medoid = real instance")
    ap.add_argument("--use-faiss", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    extra = [int(k) for k in args.exemplar_ks.split(",") if k.strip()]
    args.exemplar_ks = [args.num_cluster] + [k for k in extra if k != args.num_cluster]

    print(f"[INFO] Loading complete groups from {args.aug_groups}...")
    with open(args.aug_groups) as f:
        aug_groups = json.load(f)
    print(f"[INFO] Loaded {len(aug_groups)} complete groups")
    df = pd.read_csv(args.input).reset_index(drop=True)
    print(f"[INFO] Loaded {len(df)} instances from {args.input}")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    device = torch.device(args.device)
    encoder = BertCLSEncoder(args.model_name).to(device)
    ds = CompleteGroupsDataset(df, aug_groups, tokenizer, args.max_length, args.pair_delim)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optim = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"\n{'='*80}\nTRAINING:  L = L_Pair (margin, gamma={args.margin}) "
          f"+ {args.exemplar_weight} * L_Exem (softmax, tau={args.tau})\n{'='*80}")
    print(f"  Groups: {len(aug_groups)}   Epochs: {args.epochs} (warm-up {args.warmup_epochs})")
    print(f"  Exemplar granularities k = {args.exemplar_ks}   type = {args.exemplar_type}")

    banks = None
    encoder.train()
    for epoch in range(args.epochs):
        if epoch >= args.warmup_epochs:
            print(f"\n[INFO] Epoch {epoch+1}: refreshing K-Means exemplars on current encoder")
            _, banks = build_exemplar_bank(encoder, tokenizer, df, args, device, args.seed + epoch)

        run = defaultdict(float); steps = 0; stats = defaultdict(int)
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}" + ("" if banks else " [warm-up]"))
        for batch in pbar:
            a_ids, a_attn, a_tti = (batch[k].to(device) for k in
                                    ("a_input_ids", "a_attention_mask", "a_token_type_ids"))
            p_ids, p_attn, p_tti = (batch[k].to(device) for k in
                                    ("p_input_ids", "p_attention_mask", "p_token_type_ids"))
            B, N, L = p_ids.shape

            a = encoder(a_ids, a_attn, a_tti)
            p_flat = encoder(p_ids.view(B * N, L), p_attn.view(B * N, L), p_tti.view(B * N, L))
            p = p_flat.view(B, N, -1)

            n_emb, bs = sample_negatives(a.detach(), batch["base_id"], args.batch_size)
            for k_, v in bs.items():
                stats[k_] += v
            loss_pair = pair_margin_loss(a, p, n_emb, args.margin)

            if banks is not None:
                z = torch.cat([a, p_flat], dim=0)
                row_idx = torch.cat([batch["a_row_idx"], batch["p_row_idx"].view(-1)]).to(device)
                loss_ex = exemplar_loss(z, row_idx, banks, args.tau)
                loss = loss_pair + args.exemplar_weight * loss_ex
            else:
                loss_ex = torch.tensor(0.0)
                loss = loss_pair

            optim.zero_grad(); loss.backward(); optim.step()
            run['total'] += loss.item(); run['pair'] += loss_pair.item(); run['ex'] += float(loss_ex)
            steps += 1
            pbar.set_postfix({'L': f'{loss.item():.3f}', 'pair': f'{loss_pair.item():.3f}',
                              'exem': f'{float(loss_ex):.3f}'})

        print(f"\n[INFO] Epoch {epoch+1}: total={run['total']/steps:.4f} "
              f"pair={run['pair']/steps:.4f} exem={run['ex']/steps:.4f} "
              f"valid_neg={stats['valid']} fallback={stats['fallback']}")

    # ---- inference: plain K-Means on the trained encoder (AugURE 4.3.2, last para) ----
    print(f"\n{'='*80}\nFINAL: ENCODE -> K-MEANS (k={args.num_cluster})\n{'='*80}")
    embeddings = encode_all(encoder, tokenizer, df, args.max_length, device, args.pair_delim)
    cluster_ids, centroids = run_kmeans(embeddings, args.num_cluster, args.seed, args.use_faiss, args.gpu)
    k = args.num_cluster

    # exemplar vector per cluster and nearest real instance (for naming prompts)
    if args.exemplar_type == "centroid":
        E = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
        sims = embeddings @ E.T
        ex_idx = np.zeros(k, dtype=np.int64)
        for c in range(k):
            members = np.where(cluster_ids == c)[0]
            ex_idx[c] = members[np.argmax(sims[members, c])] if len(members) else int(np.argmax(sims[:, c]))
    else:
        ex_idx = medoid_exemplars(embeddings, cluster_ids, k)
        E = embeddings[ex_idx]
        sims = embeddings @ E.T
    dist = 1.0 - sims[np.arange(len(df)), cluster_ids]

    ex_inst = df['instance_id'].astype(str).values[ex_idx]
    df['cluster_id'] = cluster_ids
    df['exemplar_instance_id'] = ex_inst[cluster_ids]
    df['dist_to_exemplar'] = dist
    df['is_exemplar'] = False
    df.loc[ex_idx, 'is_exemplar'] = True
    df.to_csv(args.output, index=False)
    np.save(args.output_embeddings, embeddings)
    np.save(args.output_centroids, centroids)
    print(f"[INFO] ✅ Saved {args.output}, {args.output_embeddings}, {args.output_centroids}")

    sizes = Counter(cluster_ids)
    rows = []
    for c in range(k):
        r = df.iloc[ex_idx[c]]
        rows.append({'cluster_id': c, 'size': sizes.get(c, 0), 'exemplar_row': int(ex_idx[c]),
                     'exemplar_instance_id': str(r['instance_id']),
                     'predicate_lemma': r['predicate_lemma'], 'object_lemma': r['object_lemma'],
                     'sentence': r['sentence']})
    pd.DataFrame(rows).to_csv(args.output_exemplars, index=False)
    print(f"[INFO] ✅ Saved exemplar table to {args.output_exemplars}")

    print(f"\n[INFO] Cluster distribution (top 10):")
    for cid, cnt in sizes.most_common(10):
        r = df.iloc[ex_idx[cid]]
        print(f"  Cluster {cid:3d} ({cnt:5d}): {r['predicate_lemma']} | {r['object_lemma']}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
