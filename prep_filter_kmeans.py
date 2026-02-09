#!/usr/bin/env python3
"""
Filter to ONLY Complete Augmentation Groups

CRITICAL FILTERING:
- Keeps ONLY base instances with EXACTLY 5 augmentations (6 total: 1 original + 5 aug)
- Discards incomplete groups (0-4 augmentations)
- Outputs clean data for training (no padding/masking needed)

Why this matters:
- Balanced training (all instances have same number of positives)
- Simpler training code (no variable-length handling)
- Better data quality (all augmentations successful)
- Efficient computation (no wasted padding)

Output:
- aug_groups.json: ONLY groups with exactly 6 members
- aug_metadata.csv: Metadata for ALL instances (with validity flags)
- aug_validation.txt: Detailed report of what was kept/dropped
- train_aug_instances_complete.csv: ONLY instances in complete groups
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import pandas as pd


AUG_SUFFIX_RE = re.compile(r"_aug\d+$")


def is_augmented_id(instance_id: str) -> bool:
    return bool(AUG_SUFFIX_RE.search(str(instance_id)))


def parse_base_instance_id(instance_id: str) -> str:
    return AUG_SUFFIX_RE.sub("", str(instance_id))


def build_complete_groups(
    df: pd.DataFrame, 
    required_augmentations: int = 5
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict]:
    """
    Build augmentation groups, keeping ONLY those with EXACTLY required_augmentations.
    
    Args:
        df: Input dataframe with instance_id column
        required_augmentations: Number of augmentations required (default: 5)
    
    Returns:
        df_out: df with helper columns + validity flags
        complete_groups: ONLY groups with exactly (1 + required_augmentations) members
        stats: Detailed statistics
    """
    df = df.copy()
    df["instance_id"] = df["instance_id"].astype(str)
    
    df["is_augmented"] = df["instance_id"].apply(is_augmented_id)
    df["base_instance_id"] = df["instance_id"].apply(parse_base_instance_id)
    
    original_ids: Set[str] = set(df.loc[~df["is_augmented"], "instance_id"].tolist())
    
    print(f"\n[INFO] Total instances: {len(df)}")
    print(f"[INFO] Original instances: {len(original_ids)}")
    print(f"[INFO] Augmented instances: {df['is_augmented'].sum()}")
    
    df["aug_maps_to_original"] = True
    invalid_aug_mask = df["is_augmented"] & (~df["base_instance_id"].isin(original_ids))
    df.loc[invalid_aug_mask, "aug_maps_to_original"] = False
    
    invalid_aug_count = invalid_aug_mask.sum()
    if invalid_aug_count > 0:
        print(f"[WARN] Found {invalid_aug_count} augmentations without valid original - DROPPING")
    
    valid_mask = (~df["is_augmented"]) | (df["is_augmented"] & df["aug_maps_to_original"])
    df_valid = df.loc[valid_mask].copy()
    
    all_groups = defaultdict(list)
    for inst_id, base_id in zip(df_valid["instance_id"], df_valid["base_instance_id"]):
        all_groups[str(base_id)].append(str(inst_id))
    
    complete_groups: Dict[str, List[str]] = {}
    incomplete_groups = defaultdict(list)
    
    expected_size = 1 + required_augmentations
    
    for base_id, members in all_groups.items():
        members_unique = list(dict.fromkeys(members))
        
        if is_augmented_id(base_id):
            incomplete_groups["base_id_looks_augmented"].append(base_id)
            continue
        
        if base_id not in original_ids:
            incomplete_groups["base_id_not_original"].append(base_id)
            continue
        
        if base_id not in members_unique:
            incomplete_groups["original_missing_from_group"].append(base_id)
            continue
        
        if len(members_unique) != expected_size:
            num_aug = len(members_unique) - 1
            reason = f"has_{num_aug}_augmentations"
            incomplete_groups[reason].append(base_id)
            continue
        
        complete_groups[base_id] = members_unique
    
    complete_instance_ids = set()
    for members in complete_groups.values():
        complete_instance_ids.update(members)
    
    df["is_in_complete_group"] = df["instance_id"].isin(complete_instance_ids)
    
    stats = {
        "total_instances": int(len(df)),
        "original_instances": int(len(original_ids)),
        "augmented_instances": int(df["is_augmented"].sum()),
        "invalid_augmentations_dropped": int(invalid_aug_count),
        "required_augmentations": required_augmentations,
        "expected_group_size": expected_size,
        "total_groups_found": int(len(all_groups)),
        "complete_groups": int(len(complete_groups)),
        "incomplete_groups": int(len(all_groups) - len(complete_groups)),
        "instances_in_complete_groups": int(len(complete_instance_ids)),
        "instances_dropped": int(len(df) - len(complete_instance_ids)),
        "incomplete_breakdown": {k: len(v) for k, v in incomplete_groups.items()},
    }
    
    print(f"\n[INFO] Group Filtering Results:")
    print(f"  Total groups found: {stats['total_groups_found']}")
    print(f"  Complete groups (exactly {required_augmentations} aug): {stats['complete_groups']} ✓")
    print(f"  Incomplete groups: {stats['incomplete_groups']} ✗")
    
    if stats['incomplete_groups'] > 0:
        print(f"\n[INFO] Incomplete group breakdown:")
        for reason, count in sorted(stats['incomplete_breakdown'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count} groups")
    
    print(f"\n[INFO] Instance Filtering:")
    print(f"  Instances in complete groups: {stats['instances_in_complete_groups']} ✓")
    print(f"  Instances dropped: {stats['instances_dropped']} ✗")
    print(f"  Retention rate: {100 * stats['instances_in_complete_groups'] / stats['total_instances']:.1f}%")
    
    return df, complete_groups, stats


def validate_complete_groups(groups: Dict[str, List[str]], required_size: int) -> Dict:
    """
    Validate that all groups are complete and correct.
    """
    report = {
        "total_groups": len(groups),
        "required_size": required_size,
        "errors": [],
        "warnings": [],
        "is_valid": True,
    }
    
    seen_instances = set()
    
    for base_id, members in groups.items():
        if len(members) != required_size:
            report["errors"].append(
                f"Group {base_id}: expected {required_size} members, got {len(members)}"
            )
        
        if is_augmented_id(base_id):
            report["errors"].append(f"Group {base_id}: base_id looks augmented")
        
        if base_id not in members:
            report["errors"].append(f"Group {base_id}: original missing from members")
        
        if len(members) != len(set(members)):
            report["errors"].append(f"Group {base_id}: contains duplicate members")
        
        for m in members:
            if m in seen_instances:
                report["errors"].append(f"Instance {m}: appears in multiple groups")
            seen_instances.add(m)
    
    report["is_valid"] = len(report["errors"]) == 0
    
    if report["is_valid"]:
        print(f"\n[INFO] ✓ Validation PASSED: All {len(groups)} groups are complete and valid!")
    else:
        print(f"\n[ERROR] ✗ Validation FAILED: {len(report['errors'])} errors found")
    
    return report


def write_report(path: str, stats: Dict, report: Dict) -> None:
    """Write detailed validation report."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AUGMENTATION GROUP FILTERING REPORT (COMPLETE GROUPS ONLY)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write(f"  Required augmentations per instance: {stats['required_augmentations']}\n")
        f.write(f"  Expected group size: {stats['expected_group_size']} (1 original + {stats['required_augmentations']} aug)\n\n")
        
        f.write("INPUT DATA\n")
        f.write(f"  Total instances: {stats['total_instances']}\n")
        f.write(f"  Original instances: {stats['original_instances']}\n")
        f.write(f"  Augmented instances: {stats['augmented_instances']}\n\n")
        
        f.write("FILTERING RESULTS\n")
        f.write(f"  Total groups found: {stats['total_groups_found']}\n")
        f.write(f"  Complete groups (kept): {stats['complete_groups']} ✓\n")
        f.write(f"  Incomplete groups (dropped): {stats['incomplete_groups']} ✗\n\n")
        
        if stats['incomplete_breakdown']:
            f.write("INCOMPLETE GROUP BREAKDOWN\n")
            for reason, count in sorted(stats['incomplete_breakdown'].items(), key=lambda x: -x[1]):
                f.write(f"  {reason}: {count} groups\n")
            f.write("\n")
        
        f.write("INSTANCE FILTERING\n")
        f.write(f"  Instances in complete groups: {stats['instances_in_complete_groups']} ✓\n")
        f.write(f"  Instances dropped: {stats['instances_dropped']} ✗\n")
        f.write(f"  Retention rate: {100 * stats['instances_in_complete_groups'] / stats['total_instances']:.1f}%\n\n")
        
        f.write("VALIDATION\n")
        f.write(f"  Total groups validated: {report['total_groups']}\n")
        f.write(f"  Status: {'VALID ✓' if report['is_valid'] else 'ERRORS FOUND ✗'}\n\n")
        
        if report["errors"]:
            f.write("VALIDATION ERRORS\n")
            for e in report["errors"]:
                f.write(f"  - {e}\n")
            f.write("\n")
        
        if report["warnings"]:
            f.write("WARNINGS\n")
            for w in report["warnings"]:
                f.write(f"  - {w}\n")


def create_metadata(df: pd.DataFrame, complete_groups: Dict[str, List[str]], required_aug: int) -> pd.DataFrame:
    """Add metadata columns to dataframe."""
    group_sizes = {b: len(m) for b, m in complete_groups.items()}
    
    out = df.copy()
    out["group_size"] = out["base_instance_id"].map(group_sizes).fillna(0).astype(int)
    out["num_augmentations"] = (out["group_size"] - 1).clip(lower=0)
    out["is_complete_group"] = out["group_size"] == (1 + required_aug)
    
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Filter augmentation groups to ONLY complete ones"
    )
    
    ap.add_argument(
        "--input",
        default="data/train_aug_instances.csv",
        help="Input CSV with all instances"
    )
    ap.add_argument(
        "--required-augmentations",
        type=int,
        default=5,
        help="Required number of augmentations per instance (default: 5)"
    )
    ap.add_argument(
        "--output-groups",
        default="data/aug_groups.json",
        help="Output: Complete groups only"
    )
    ap.add_argument(
        "--output-metadata",
        default="data/aug_metadata.csv",
        help="Output: Metadata for all instances (with flags)"
    )
    ap.add_argument(
        "--output-report",
        default="data/aug_validation.txt",
        help="Output: Validation report"
    )
    ap.add_argument(
        "--output-complete-csv",
        default="data/train_aug_instances_complete.csv",
        help="Output: CSV with ONLY instances in complete groups"
    )
    
    args = ap.parse_args()
    
    print(f"[INFO] Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    if "instance_id" not in df.columns:
        raise ValueError("Input CSV must have 'instance_id' column")
    
    df_meta, complete_groups, stats = build_complete_groups(df, args.required_augmentations)
    
    report = validate_complete_groups(complete_groups, 1 + args.required_augmentations)
    
    os.makedirs(os.path.dirname(args.output_groups) or ".", exist_ok=True)
    with open(args.output_groups, "w") as f:
        json.dump(complete_groups, f, indent=2)
    print(f"\n[INFO] Saved complete groups to {args.output_groups}")
    
    meta = create_metadata(df_meta, complete_groups, args.required_augmentations)
    os.makedirs(os.path.dirname(args.output_metadata) or ".", exist_ok=True)
    meta.to_csv(args.output_metadata, index=False)
    print(f"[INFO] Saved metadata to {args.output_metadata}")
    
    write_report(args.output_report, stats, report)
    print(f"[INFO] Saved report to {args.output_report}")
    
    complete_df = df_meta[df_meta["is_in_complete_group"]].copy()
    os.makedirs(os.path.dirname(args.output_complete_csv) or ".", exist_ok=True)
    complete_df.to_csv(args.output_complete_csv, index=False)
    print(f"[INFO] Saved complete instances CSV to {args.output_complete_csv}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Complete groups: {len(complete_groups)}")
    print(f"Complete instances: {len(complete_df)} (out of {len(df)} total)")
    print(f"Retention: {100 * len(complete_df) / len(df):.1f}%")
    print(f"Status: {'✓ VALID' if report['is_valid'] else '✗ ERRORS FOUND'}")
    
    if not report["is_valid"]:
        print(f"\n✗ Validation failed! See {args.output_report} for details")
        raise SystemExit(1)
    
    print(f"\n✓ Ready for training! Use {args.output_complete_csv}")


if __name__ == "__main__":
    main()
