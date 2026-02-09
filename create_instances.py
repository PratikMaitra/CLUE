#!/usr/bin/env python3


import csv
import argparse
from collections import OrderedDict, Counter


def create_instances(input_path: str, output_path: str):
    
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"[INFO] Read {len(rows)} rows from {input_path}")
    
    unique_pairs = OrderedDict()
    
    for row in rows:
        predicate = row.get('predicate_lemma', '')
        obj = row.get('object_lemma', '')
        
        pair_key = (predicate, obj)
        
        if pair_key not in unique_pairs:
            unique_pairs[pair_key] = row
    
    print(f"[INFO] Found {len(unique_pairs)} unique (predicate, object) pairs")
    print(f"[INFO] Removed {len(rows) - len(unique_pairs)} duplicate pairs")
    
    sent_id_to_pairs = {}
    for (predicate, obj), row in unique_pairs.items():
        sent_id = row.get('sent_id', row.get('wnd_id', ''))
        if sent_id not in sent_id_to_pairs:
            sent_id_to_pairs[sent_id] = []
        sent_id_to_pairs[sent_id].append(((predicate, obj), row))
    
    output_rows = []
    
    for (predicate, obj), row in unique_pairs.items():
        sentence = row.get('sentence', '')
        sent_id = row.get('sent_id', row.get('wnd_id', ''))
        
        pairs_in_sent = sent_id_to_pairs[sent_id]
        pair_number = next(i for i, (p, r) in enumerate(pairs_in_sent, 1) if p == (predicate, obj))
        
        instance = f"⟨{predicate} | {obj}, {sentence}⟩"
        
        instance_id = f"{sent_id}_insta{pair_number}"
        
        new_row = row.copy()
        new_row['instance_id'] = instance_id
        new_row['instance'] = instance
        
        output_rows.append(new_row)
    
    output_fieldnames = ['instance_id', 'instance'] + list(fieldnames)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"[INFO] Created {len(output_rows)} instances")
    print(f"[INFO] Output written to: {output_path}")
    
    print("\n" + "="*80)
    print("SAMPLE INSTANCES")
    print("="*80)
    for i, row in enumerate(output_rows[:5], 1):
        print(f"\nInstance {i}:")
        print(f"  ID: {row['instance_id']}")
        pred = row.get('predicate_lemma', '')
        obj = row.get('object_lemma', '')
        print(f"  Pair: ({pred}, {obj})")
        instance = row['instance']
        print(f"  Instance: {instance[:100]}..." if len(instance) > 100 else f"  Instance: {instance}")
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    unique_predicates = set(p for p, o in unique_pairs.keys())
    unique_objects = set(o for p, o in unique_pairs.keys())
    
    print(f"\nUnique predicates: {len(unique_predicates)}")
    print(f"Unique objects: {len(unique_objects)}")
    print(f"Unique (predicate, object) pairs: {len(unique_pairs)}")
    
    pred_counts = Counter(p for p, o in unique_pairs.keys())
    print(f"\nTop 10 predicates:")
    for pred, count in pred_counts.most_common(10):
        print(f"  {pred}: {count} unique pair(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Create instances from unique predicate-object pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Instance format: ⟨predicate | object, sentence⟩

Instance ID format: {sent_id}_insta{number}

Duplicates: If the same (predicate, object) pair appears in multiple sentences,
only the first occurrence is kept.
        """
    )
    
    parser.add_argument(
        '--input',
        default='data/train_predicate_object.csv',
        help='Input CSV file (default: train_predicate_object.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/train_instances.csv',
        help='Output CSV file (default: train_instances.csv)'
    )
    
    args = parser.parse_args()
    
    create_instances(args.input, args.output)


if __name__ == '__main__':
    main()
