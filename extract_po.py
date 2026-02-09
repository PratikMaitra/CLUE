#!/usr/bin/env python3


import os
import re
import json
import csv
import argparse
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from tqdm import tqdm
from stanza.server import CoreNLPClient




STOP_WORDS = {
    "the", "to", "and", "a", "an", "in", "it", "is", "are", "of", "i", "that",
    "had", "on", "for", "were", "was", "from", "by", "with", "have", "has", "be",
    "as", "at", "this", "these", "those"
}

LIGHT_VERB_LEMMAS = {"do", "be", "have", "get", "say"}
TEMPORAL_WORDS = {"pm", "am", "gmt", "time", "hour", "minute", "second", "o'clock"}

_NUM_RE = re.compile(r"^\d+([.,]\d+)?$")


def is_valid_predicate(lemma: str, pos: str) -> bool:
    
    if not pos.startswith("VB"):
        return False
    
    lemma_lower = lemma.lower()
    
    if lemma_lower in LIGHT_VERB_LEMMAS:
        return False
    if lemma_lower in STOP_WORDS:
        return False
    if len(lemma_lower) <= 1:
        return False
    
    return True


def is_valid_object(lemma: str, pos: str) -> bool:
    
    if not (pos.startswith("NN") or pos in {"PRP", "PRP$"}):
        return False
    
    lemma_lower = lemma.lower()
    
    if lemma_lower in TEMPORAL_WORDS:
        return False
    
    if _NUM_RE.match(lemma_lower) or pos == "CD":
        return False
    
    if lemma_lower in STOP_WORDS or len(lemma_lower) <= 1:
        return False
    
    return True


def get_gold_event_types(obj: dict, use_coarse: bool = True, coarse_separator: str = ':') -> Set[str]:
   
    event_types = set()
    
    event_mentions = obj.get("event_mentions", []) or []
    for em in event_mentions:
        event_type = em.get("event_type", "")
        if event_type:
            if use_coarse and coarse_separator in event_type:
                event_type = event_type.split(coarse_separator)[0]
            event_types.add(event_type)
    
    return event_types


def extract_from_openie(sentence_json: dict) -> Set[Tuple[str, str, str, int]]:
    
    pairs = set()
    
    tokens = sentence_json.get("tokens", [])
    openie_triples = sentence_json.get("openie", [])
    
    for triple in openie_triples:
        rel_span = triple.get("relationSpan", [])
        obj_span = triple.get("objectSpan", [])
        
        if not rel_span or len(rel_span) < 2:
            continue
        if not obj_span or len(obj_span) < 2:
            continue
        
        pred_lemma = None
        pred_word = None
        pred_index = None
        
        for i in range(rel_span[0], min(rel_span[1], len(tokens))):
            tok = tokens[i]
            pos = tok.get("pos", "")
            if pos.startswith("VB"):
                lemma = tok.get("lemma", "")
                if is_valid_predicate(lemma, pos):
                    pred_lemma = lemma.lower()
                    pred_word = tok.get("word", "")
                    pred_index = i
                    break
        
        if not pred_lemma:
            continue
        
        obj_lemma = None
        for i in range(obj_span[0], min(obj_span[1], len(tokens))):
            tok = tokens[i]
            lemma = tok.get("lemma", "")
            pos = tok.get("pos", "")
            if is_valid_object(lemma, pos):
                obj_lemma = lemma.lower()
                break
        
        if not obj_lemma:
            continue
        
        pairs.add((pred_lemma, obj_lemma, pred_word, pred_index))
    
    return pairs


def extract_from_dependencies(sentence_json: dict) -> Set[Tuple[str, str, str, int]]:
    """Extract (predicate_lemma, object_lemma, predicate_word, predicate_index) from dependencies."""
    pairs = set()
    
    tokens = sentence_json.get("tokens", [])
    deps = sentence_json.get("basicDependencies", [])
    if not deps:
        deps = sentence_json.get("enhancedDependencies", [])
    
    for dep in deps:
        rel = dep.get("dep", "")
        gov_idx = dep.get("governor", 0)
        dep_idx = dep.get("dependent", 0)
        
        if rel in {"dobj", "obj", "iobj", "nsubjpass"}:
            if 0 < gov_idx <= len(tokens) and 0 < dep_idx <= len(tokens):
                gov_tok = tokens[gov_idx - 1]
                dep_tok = tokens[dep_idx - 1]
                
                pred_lemma = gov_tok.get("lemma", "")
                pred_pos = gov_tok.get("pos", "")
                obj_lemma = dep_tok.get("lemma", "")
                obj_pos = dep_tok.get("pos", "")
                
                if is_valid_predicate(pred_lemma, pred_pos) and is_valid_object(obj_lemma, obj_pos):
                    pred_word = gov_tok.get("word", "")
                    pred_index = gov_idx - 1
                    pairs.add((pred_lemma.lower(), obj_lemma.lower(), pred_word, pred_index))
    
    return pairs


def extract_unique_predicates(sentence_json: dict) -> Dict[str, Tuple[str, str, int]]:
  
    openie_pairs = extract_from_openie(sentence_json)
    dep_pairs = extract_from_dependencies(sentence_json)
    
    all_pairs = openie_pairs | dep_pairs
    
    predicate_to_info = {}
    for pred_lemma, obj_lemma, pred_word, pred_index in all_pairs:
        if pred_lemma not in predicate_to_info:
            predicate_to_info[pred_lemma] = (obj_lemma, pred_word, pred_index)
    
    return predicate_to_info


def iter_jsonl(path: str):
    """Iterate over JSONL file, yielding parsed objects."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            yield obj


def main():
    parser = argparse.ArgumentParser(
        description="Extract predicate-object pairs WITH gold event types",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", help="Input JSONL file", default="data/train.json")
    parser.add_argument("--output", help="Output CSV file", default="data/train_predicate_object.csv")
    parser.add_argument("--memory", default="32G", help="JVM memory")
    parser.add_argument("--timeout_ms", type=int, default=600000)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--use-coarse", action="store_true", default=True,
                       help="Extract coarse-grained event types (default: True)")
    parser.add_argument("--no-coarse", action="store_false", dest="use_coarse",
                       help="Use fine-grained event types")
    parser.add_argument("--coarse-separator", default=":",
                       help="Separator for coarse/fine types (default: ':')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    annotators = ["tokenize", "ssplit", "pos", "lemma", "depparse", "natlog", "openie"]
    
    all_rows = []
    total_records = 0
    
    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Output: {args.output}")
    print(f"[INFO] Use coarse types: {args.use_coarse}")
    print("[INFO] Starting CoreNLP client...")
    
    with CoreNLPClient(
        annotators=annotators,
        timeout=args.timeout_ms,
        memory=args.memory,
        be_quiet=True,
        start_server=True,
    ) as client:
        
        for rec in tqdm(iter_jsonl(args.input), desc="Processing"):
            total_records += 1
            if args.max_records and total_records > args.max_records:
                break
            
            text = rec.get("text", "")
            doc_id = rec.get("doc_id", "")
            wnd_id = rec.get("wnd_id", "")
            
            gold_types = get_gold_event_types(rec, args.use_coarse, args.coarse_separator)
            gold_types_str = ",".join(sorted(gold_types)) if gold_types else ""
            
            try:
                ann = client.annotate(text, output_format="json")
            except Exception as e:
                print(f"[WARN] Failed: {e}")
                continue
            
            for sent_idx, sent_json in enumerate(ann.get("sentences", []) or []):
                predicate_to_info = extract_unique_predicates(sent_json)
                
                for pred_lemma, (obj_lemma, pred_word, pred_index) in predicate_to_info.items():
                    all_rows.append({
                        "doc_id": doc_id,
                        "sent_id": wnd_id,
                        "sentence": text,
                        "sent_index": sent_idx,
                        "predicate_lemma": pred_lemma,
                        "object_lemma": obj_lemma,
                        "predicate": pred_word,
                        "predicate_index": pred_index,
                        "gold_event_types": gold_types_str,  
                    })
    
    if all_rows:
        fieldnames = ["doc_id", "sent_id", "sentence", "sent_index", "predicate_lemma", 
                     "object_lemma", "predicate", "predicate_index", "gold_event_types"]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    
    unique_predicates = set(row["predicate_lemma"] for row in all_rows)
    rows_with_gold = sum(1 for row in all_rows if row["gold_event_types"])
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Records processed:         {total_records}")
    print(f"Total instances:           {len(all_rows)}")
    print(f"Instances with gold types: {rows_with_gold} ({rows_with_gold/len(all_rows)*100:.1f}%)")
    print(f"Unique predicates:         {len(unique_predicates)}")
    print(f"Avg instances/sentence:    {len(all_rows) / max(total_records, 1):.2f}")
    print(f"{'='*60}")
    print(f"\nOutput: {args.output}")
    
    if all_rows:
        from collections import Counter
        pred_counts = Counter(row["predicate_lemma"] for row in all_rows)
        print(f"\nTop 10 predicates:")
        for pred, count in pred_counts.most_common(10):
            print(f"  {pred}: {count}")
        
        print(f"\nSample instances with gold types:")
        samples_with_gold = [r for r in all_rows if r["gold_event_types"]][:5]
        for row in samples_with_gold:
            print(f"  {row['predicate']}: {row['gold_event_types']}")


if __name__ == "__main__":
    main()
