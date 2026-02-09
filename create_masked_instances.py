#!/usr/bin/env python3
"""
Create masked instances from train_instances.csv

Adds a new column 'masked_instance' where the predicate in the sentence is replaced with [MASK].

Format: ⟨p | o, S_masked⟩
Where S_masked = S[1:i] ⊕ [MASK] ⊕ S[j:n]

Input: data/train_instances.csv
Output: data/train_masked_instances.csv

Usage:
  python create_masked_instances.py --input data/train_instances.csv --output data/train_masked_instances.csv
"""

import csv
import argparse
import re


def find_predicate_character_positions(sentence: str, predicate_text: str, predicate_index: int) -> tuple:
    """
    Find the character start and end positions of the predicate in the sentence
    using the token index.
    
    Args:
        sentence: The full sentence text
        predicate_text: The predicate word to find
        predicate_index: The token index of the predicate (0-based)
    
    Returns:
        (start_pos, end_pos) tuple in character positions, or None if not found
    """
    current_char_pos = 0
    current_token_index = 0
    
    in_word = False
    word_start = 0
    word = ""
    
    for i, char in enumerate(sentence):
        if char.isalnum() or char in "'-":
            if not in_word:
                word_start = i
                word = char
                in_word = True
            else:
                word += char
        else:
            if in_word:
                word_end = i
                
                if current_token_index == predicate_index:
                    if word.lower() == predicate_text.lower() or predicate_text.lower() in word.lower():
                        return (word_start, word_end)
                
                current_token_index += 1
                in_word = False
                word = ""
    
    if in_word:
        if current_token_index == predicate_index:
            if word.lower() == predicate_text.lower() or predicate_text.lower() in word.lower():
                return (word_start, len(sentence))
    
    pattern = r'\b' + re.escape(predicate_text) + r'\b'
    matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
    
    if matches:
        if predicate_index < len(matches):
            return (matches[predicate_index].start(), matches[predicate_index].end())
        else:
            return (matches[0].start(), matches[0].end())
    
    return None


def create_masked_sentence(sentence: str, start_pos: int, end_pos: int) -> str:
    """
    Create masked sentence: S[1:i] ⊕ [MASK] ⊕ [j:n]
    
    Args:
        sentence: Original sentence
        start_pos: Start character position of predicate (i)
        end_pos: End character position of predicate (j)
    
    Returns:
        Masked sentence with [MASK] replacing the predicate
    """
    before = sentence[:start_pos]
    after = sentence[end_pos:]
    
    masked_sentence = before + "[MASK]" + after
    
    return masked_sentence


def create_masked_instance(instance_row: dict) -> str:
    """
    Create masked instance: ⟨p | o, S_masked⟩
    
    Args:
        instance_row: Dictionary containing instance data
    
    Returns:
        Masked instance string
    """
    predicate_lemma = instance_row.get('predicate_lemma', '')
    obj_lemma = instance_row.get('object_lemma', '')
    sentence = instance_row.get('sentence', '')
    
    predicate_text = instance_row.get('predicate', predicate_lemma)
    predicate_index = instance_row.get('predicate_index')
    
    if predicate_index is not None:
        try:
            predicate_index = int(predicate_index)
        except:
            predicate_index = None
    
    positions = find_predicate_character_positions(sentence, predicate_text, predicate_index)
    
    if positions is None:
        masked_sentence = sentence
        print(f"[WARN] Could not find predicate '{predicate_text}' at index {predicate_index} in sentence: {sentence[:60]}...")
    else:
        start_pos, end_pos = positions
        masked_sentence = create_masked_sentence(sentence, start_pos, end_pos)
    
    masked_instance = f"⟨{predicate_lemma} | {obj_lemma}, {masked_sentence}⟩"
    
    return masked_instance


def process_instances(input_path: str, output_path: str):
    """
    Process train_instances.csv and add masked_instance column.
    
    Args:
        input_path: Path to input CSV (train_instances.csv)
        output_path: Path to output CSV (train_masked_instances.csv)
    """
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)
    
    print(f"[INFO] Read {len(rows)} rows from {input_path}")
    
    output_rows = []
    warnings = 0
    
    for i, row in enumerate(rows, 1):
        masked_instance = create_masked_instance(row)
        
        if '[MASK]' not in masked_instance:
            warnings += 1
        
        new_row = row.copy()
        new_row['masked_instance'] = masked_instance
        
        output_rows.append(new_row)
        
        if i % 1000 == 0:
            print(f"[INFO] Processed {i}/{len(rows)} rows...")
    
    output_fieldnames = fieldnames.copy()
    if 'masked_instance' not in output_fieldnames:
        output_fieldnames.append('masked_instance')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"[INFO] Wrote {len(output_rows)} rows to {output_path}")
    
    if warnings > 0:
        print(f"[WARN] {warnings} rows had issues with masking (predicate not found)")
    
    print("\n" + "="*80)
    print("SAMPLE MASKED INSTANCES")
    print("="*80)
    
    for i, row in enumerate(output_rows[:5], 1):
        print(f"\nInstance {i}:")
        print(f"  Predicate: '{row.get('predicate', '')}' at index {row.get('predicate_index', '')}")
        print(f"  Original instance: {row.get('instance', '')[:80]}...")
        print(f"  Masked instance:   {row['masked_instance'][:80]}...")
        
        if '[MASK]' in row['masked_instance']:
            masked_sent = row['masked_instance'].split(', ', 1)[1].rstrip('⟩') if ', ' in row['masked_instance'] else ''
            pred = row.get('predicate', '')
            if pred and pred not in masked_sent:
                print(f"  ✓ Masking successful - '{pred}' replaced with [MASK]")
            elif '[MASK]' in masked_sent:
                print(f"  ✓ Masking successful")
            else:
                print(f"  ⚠ Warning: [MASK] present but check needed")
        else:
            print(f"  ✗ Warning: No [MASK] found")
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    total_masked = sum(1 for row in output_rows if '[MASK]' in row['masked_instance'])
    
    print(f"\nTotal instances: {len(output_rows)}")
    print(f"Successfully masked: {total_masked}")
    print(f"Masking rate: {total_masked/len(output_rows)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Create masked instances by replacing predicates with [MASK]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates masked instances where the predicate is replaced with [MASK].

Masked instance format: ⟨p | o, S_masked⟩
Where S_masked = S[1:i] ⊕ [MASK] ⊕ S[j:n]

The script uses predicate_index to locate the predicate and replace it with [MASK].
All input columns are preserved in output.
        """
    )
    
    parser.add_argument(
        '--input',
        default='data/train_instances.csv',
        help='Input CSV file (default: data/train_instances.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/train_masked_instances.csv',
        help='Output CSV file (default: data/train_masked_instances.csv)'
    )
    
    args = parser.parse_args()
    
    process_instances(args.input, args.output)


if __name__ == '__main__':
    main()
