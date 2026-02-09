#!/usr/bin/env python3


import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Apply cluster merge mapping to clusters.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        '--clusters',
        default='data/clusters.csv',
        help='Input clusters file (default: data/clusters.csv)'
    )
    parser.add_argument(
        '--mapping',
        default='data/cluster_names_merged.csv',
        help='Cluster merge mapping file (default: data/cluster_names_merged.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/clusters_merged.csv',
        help='Output file (default: data/clusters_merged.csv)'
    )
    
    args = parser.parse_args()
    
    # Load clusters.csv
    print(f"[INFO] Reading {args.clusters}...")
    clusters_df = pd.read_csv(args.clusters)
    print(f"[INFO] Loaded {len(clusters_df)} instances")
    print(f"[INFO] Columns: {list(clusters_df.columns)}")
    
    # Load cluster merge mapping
    print(f"\n[INFO] Reading {args.mapping}...")
    mapping_df = pd.read_csv(args.mapping)
    print(f"[INFO] Loaded mapping for {len(mapping_df)} clusters")
    
    # Verify required columns in mapping
    required_cols = ['cluster_id', 'cluster_merged_id', 'cluster_merged_name']
    missing_cols = [col for col in required_cols if col not in mapping_df.columns]
    if missing_cols:
        raise ValueError(f"Mapping file missing required columns: {missing_cols}")
    
    # Create mapping dictionaries
    print(f"\n[INFO] Creating mapping dictionaries...")
    cluster_to_merged_id = dict(zip(mapping_df['cluster_id'], mapping_df['cluster_merged_id']))
    cluster_to_merged_name = dict(zip(mapping_df['cluster_id'], mapping_df['cluster_merged_name']))
    
    print(f"[INFO] Mapping contains {len(cluster_to_merged_id)} cluster IDs")
    
    # Add cluster_merged_id and cluster_merged_name columns
    print(f"\n[INFO] Adding cluster_merged_id and cluster_merged_name columns...")
    clusters_df['cluster_merged_id'] = clusters_df['cluster_id'].map(cluster_to_merged_id)
    clusters_df['cluster_merged_name'] = clusters_df['cluster_id'].map(cluster_to_merged_name)
    
    # Check for any unmapped clusters
    unmapped = clusters_df['cluster_merged_id'].isna().sum()
    if unmapped > 0:
        print(f"[WARN] {unmapped} instances have unmapped cluster_ids")
        print(f"[WARN] These will have NaN in cluster_merged_id and cluster_merged_name")
        
        # Show some examples
        unmapped_clusters = clusters_df[clusters_df['cluster_merged_id'].isna()]['cluster_id'].unique()
        print(f"[WARN] Unmapped cluster IDs: {unmapped_clusters[:10]}")
    else:
        print(f"[INFO] All instances successfully mapped!")
    
    # Save output
    print(f"\n[INFO] Saving output to {args.output}...")
    clusters_df.to_csv(args.output, index=False)
    print(f"[INFO] Saved {len(clusters_df)} instances")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total instances: {len(clusters_df)}")
    print(f"Original unique clusters: {clusters_df['cluster_id'].nunique()}")
    print(f"Merged unique clusters: {clusters_df['cluster_merged_id'].nunique()}")
    print(f"Reduction: {clusters_df['cluster_id'].nunique() - clusters_df['cluster_merged_id'].nunique()} clusters merged")
    print(f"Reduction rate: {(1 - clusters_df['cluster_merged_id'].nunique() / clusters_df['cluster_id'].nunique()) * 100:.1f}%")
    
    # Show distribution
    print(f"\n[INFO] Top 5 merged clusters by instance count:")
    merged_counts = clusters_df['cluster_merged_name'].value_counts().head(5)
    for name, count in merged_counts.items():
        print(f"  {name}: {count} instances")
    
    # Show merge examples
    print(f"\n[INFO] Merge examples (instances that changed cluster):")
    changed = clusters_df[clusters_df['cluster_id'] != clusters_df['cluster_merged_id']].head(5)
    for _, row in changed.iterrows():
        print(f"  Instance {row.get('instance_id', 'N/A')}: "
              f"Cluster {row['cluster_id']} -> {row['cluster_merged_id']} "
              f"({row['cluster_merged_name']})")
    
    print(f"\n[INFO] Done!")
    print(f"[INFO] Output columns: {list(clusters_df.columns)}")


if __name__ == "__main__":
    main()