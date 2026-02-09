#!/usr/bin/env python3
"""
Create augmented instances using BERT masked language model.

Reads train_masked_instances.csv and generates up to 5 augmented predicates using BERT.
Creates new instances by replacing [MASK] with predicted verbs.

Input format to BERT:
  [CLS] p | o [SEP] S_masked [SEP]

Output: train_aug_instances.csv
  - Contains original instances + augmented instances
  - Augmented instances have IDs: {original_id}_aug1, {original_id}_aug2, etc.

Usage:
  python create_aug_instances.py --input data/train_masked_instances.csv --output data/train_aug_instances.csv
"""

import csv
import argparse
from typing import List, Tuple, Set
import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("[INFO] Downloading NLTK WordNet data...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def load_bert_model():
    """
    Load BERT base model and tokenizer for masked language modeling.
    
    Returns:
        tokenizer, model, device, lemmatizer
    """
    print("[INFO] Loading BERT base model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    lemmatizer = WordNetLemmatizer()
    
    print(f"[INFO] Model loaded on {device}")
    
    return tokenizer, model, device, lemmatizer


def lemmatize_verb(word: str, lemmatizer) -> str:
    """
    Lemmatize a verb to its base form.
    
    Args:
        word: The verb to lemmatize
        lemmatizer: WordNetLemmatizer instance
    
    Returns:
        Lemmatized form of the verb
    """
    return lemmatizer.lemmatize(word.lower(), pos='v')


def predict_masked_predicates(
    predicate_lemma: str,
    object_lemma: str,
    masked_sentence: str,
    tokenizer,
    model,
    device,
    top_k: int = 5,
    max_length: int = 512
) -> List[Tuple[str, float]]:
    """
    Use BERT to predict top-k verbs to replace [MASK] in the sentence.
    
    Input format: [CLS] p | o [SEP] S_masked [SEP]
    
    Args:
        predicate_lemma: Predicate lemma
        object_lemma: Object lemma
        masked_sentence: Sentence with [MASK]
        tokenizer: BERT tokenizer
        model: BERT model
        device: torch device
        top_k: Number of predictions to return (max 5)
        max_length: Maximum sequence length (default: 512)
    
    Returns:
        List of (predicted_verb, probability) tuples
    """
    bert_input = f"[CLS] {predicate_lemma} | {object_lemma} [SEP] {masked_sentence} [SEP]"
    
    tokens = tokenizer.tokenize(bert_input)
    
    try:
        mask_idx = tokens.index('[MASK]')
    except ValueError:
        return []
    
    if len(tokens) > max_length:
        sep_indices = [i for i, tok in enumerate(tokens) if tok == '[SEP]']
        
        if len(sep_indices) < 1:
            return []
        
        first_sep_idx = sep_indices[0]
        
        prefix_length = first_sep_idx + 1
        available_for_sentence = max_length - prefix_length - 1
        
        if available_for_sentence <= 0:
            return []
        
        sentence_start = first_sep_idx + 1
        sentence_tokens = tokens[sentence_start:-1]
        
        try:
            mask_pos_in_sentence = sentence_tokens.index('[MASK]')
        except ValueError:
            return []
        
        if len(sentence_tokens) > available_for_sentence:
            context_window = available_for_sentence - 1
            before_mask = context_window // 2
            after_mask = context_window - before_mask
            
            start_idx = max(0, mask_pos_in_sentence - before_mask)
            end_idx = min(len(sentence_tokens), mask_pos_in_sentence + after_mask + 1)
            
            if start_idx == 0:
                end_idx = min(len(sentence_tokens), available_for_sentence)
            elif end_idx == len(sentence_tokens):
                start_idx = max(0, len(sentence_tokens) - available_for_sentence)
            
            truncated_sentence = sentence_tokens[start_idx:end_idx]
            
            try:
                new_mask_pos = truncated_sentence.index('[MASK]')
                mask_idx = prefix_length + new_mask_pos
            except ValueError:
                return []
            
            tokens = tokens[:first_sep_idx + 1] + truncated_sentence + ['[SEP]']
        
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    if mask_idx >= len(tokens):
        return []
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs.logits
    
    mask_predictions = predictions[0, mask_idx]
    
    top_k_probs, top_k_indices = torch.topk(mask_predictions, k=top_k * 3)
    
    top_k_probs_normalized = torch.softmax(top_k_probs, dim=0)
    
    results = []
    for i, (prob, idx) in enumerate(zip(top_k_probs_normalized, top_k_indices)):
        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
        probability = prob.item()
        
        if token.startswith('##') or not token.isalpha():
            continue
        
        results.append((token, probability))
        
        if len(results) >= top_k:
            break
    
    return results[:top_k]


def create_augmented_instance(
    original_row: dict,
    augmented_verb: str,
    augmented_verb_lemma: str,
    aug_number: int,
    masked_sentence: str
) -> dict:
    """
    Create an augmented instance by replacing [MASK] with the predicted verb.
    
    Args:
        original_row: Original instance row
        augmented_verb: Predicted verb (e.g., "ripped")
        augmented_verb_lemma: Lemma of predicted verb (e.g., "rip")
        aug_number: Augmentation number (1, 2, 3, ...)
        masked_sentence: Sentence with [MASK]
    
    Returns:
        New row dict for the augmented instance
    """
    new_row = original_row.copy()
    
    original_id = original_row['instance_id']
    new_id = f"{original_id}_aug{aug_number}"
    new_row['instance_id'] = new_id
    
    new_row['predicate'] = augmented_verb
    new_row['predicate_lemma'] = augmented_verb_lemma
    
    augmented_sentence = masked_sentence.replace('[MASK]', augmented_verb)
    
    object_lemma = original_row.get('object_lemma', '')
    new_instance = f"⟨{augmented_verb_lemma} | {object_lemma}, {augmented_sentence}⟩"
    new_row['instance'] = new_instance
    
    return new_row


def process_instances(input_path: str, output_path: str, max_augmentations: int = 5):
    """
    Process instances and generate augmented versions using BERT.
    
    Args:
        input_path: Path to input CSV (train_masked_instances.csv)
        output_path: Path to output CSV (train_aug_instances.csv)
        max_augmentations: Maximum augmentations per instance (default: 5)
    """
    tokenizer, model, device, lemmatizer = load_bert_model()
    
    print(f"[INFO] Reading {input_path}...")
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)
    
    print(f"[INFO] Read {len(rows)} instances")
    
    output_rows = []
    total_augmentations = 0
    failed_augmentations = 0
    skipped_duplicates = 0
    truncated_sequences = 0
    
    for i, row in enumerate(tqdm(rows, desc="Generating augmentations"), 1):
        output_rows.append(row)
        
        masked_instance = row.get('masked_instance', '')
        
        if ', ' not in masked_instance:
            failed_augmentations += 1
            continue
        
        parts = masked_instance.split(', ', 1)
        if len(parts) != 2:
            failed_augmentations += 1
            continue
        
        masked_sentence = parts[1].rstrip('⟩')
        
        if '[MASK]' not in masked_sentence:
            failed_augmentations += 1
            continue
        
        predicate_lemma = row.get('predicate_lemma', '')
        object_lemma = row.get('object_lemma', '')
        
        seen_lemmas: Set[str] = {predicate_lemma}
        
        try:
            predictions = predict_masked_predicates(
                predicate_lemma,
                object_lemma,
                masked_sentence,
                tokenizer,
                model,
                device,
                top_k=max_augmentations * 2
            )
            
            bert_input = f"[CLS] {predicate_lemma} | {object_lemma} [SEP] {masked_sentence} [SEP]"
            tokens = tokenizer.tokenize(bert_input)
            if len(tokens) > 512:
                truncated_sequences += 1
                
        except Exception as e:
            print(f"[WARN] Row {i}: BERT prediction failed: {e}")
            failed_augmentations += 1
            continue
        
        aug_num = 0
        for predicted_verb, prob in predictions:
            predicted_verb_lemma = lemmatize_verb(predicted_verb, lemmatizer)
            
            if predicted_verb_lemma in seen_lemmas:
                skipped_duplicates += 1
                continue
            
            seen_lemmas.add(predicted_verb_lemma)
            
            aug_num += 1
            aug_row = create_augmented_instance(
                row,
                predicted_verb,
                predicted_verb_lemma,
                aug_num,
                masked_sentence
            )
            output_rows.append(aug_row)
            total_augmentations += 1
            
            if aug_num >= max_augmentations:
                break
    
    print(f"\n[INFO] Writing output to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Original instances: {len(rows)}")
    print(f"Augmented instances: {total_augmentations}")
    print(f"Skipped duplicates (same lemma): {skipped_duplicates}")
    print(f"Truncated sequences (>512 tokens): {truncated_sequences}")
    print(f"Total instances (original + augmented): {len(output_rows)}")
    print(f"Avg augmentations per original: {total_augmentations / len(rows):.2f}")
    print(f"Failed augmentations: {failed_augmentations}")
    print(f"\nOutput: {output_path}")
    
    print("\n" + "="*80)
    print("SAMPLE AUGMENTED INSTANCES")
    print("="*80)
    
    sample_id = None
    for row in output_rows:
        if not row['instance_id'].endswith(('_aug1', '_aug2', '_aug3', '_aug4', '_aug5')):
            sample_id = row['instance_id']
            break
    
    if sample_id:
        print(f"\nShowing augmentations for instance: {sample_id}\n")
        
        for row in output_rows:
            if row['instance_id'] == sample_id:
                print(f"ORIGINAL ({sample_id}):")
                print(f"  Instance: {row['instance'][:100]}...")
                print(f"  Predicate: {row.get('predicate', '')} (lemma: {row.get('predicate_lemma', '')})")
                break
        
        print(f"\nAUGMENTATIONS:")
        aug_count = 0
        for row in output_rows:
            if row['instance_id'].startswith(sample_id + '_aug'):
                aug_count += 1
                print(f"\n{row['instance_id']}:")
                print(f"  Instance: {row['instance'][:100]}...")
                print(f"  Predicate: {row.get('predicate', '')} (lemma: {row.get('predicate_lemma', '')})")
                
                if aug_count >= 5:
                    break


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented instances using BERT masked language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generates augmented predicate verbs using BERT and creates new instances.

Process:
1. Read train_masked_instances.csv
2. For each instance with [MASK]:
   - Send to BERT: [CLS] p | o [SEP] S_masked [SEP]
   - Get top 5 verb predictions
   - Create augmented instances by replacing [MASK]
3. Output original + augmented instances to train_aug_instances.csv

Augmented instances use predicted verbs with different lemmas from the original.

Requirements:
  pip install torch transformers nltk
        """
    )
    
    parser.add_argument(
        '--input',
        default='data/train_masked_instances.csv',
        help='Input CSV file (default: data/train_masked_instances.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/train_aug_instances.csv',
        help='Output CSV file (default: data/train_aug_instances.csv)'
    )
    parser.add_argument(
        '--max_aug',
        type=int,
        default=5,
        help='Maximum augmentations per instance (default: 5)'
    )
    
    args = parser.parse_args()
    
    process_instances(args.input, args.output, args.max_aug)


if __name__ == '__main__':
    main()
