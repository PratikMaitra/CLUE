# CLUE: A Contrastive Learning Framework for Unsupervised Event Type Induction and Annotation

CLUE is an unsupervised framework for event-type induction, event-type naming, and sentence-level event annotation.

<p align="center">
  <img src="assets/clue_overview.png"
       alt="Overview of the CLUE framework"
       width="900">
</p>

<p align="center">
  <em>
    Overview of CLUE: predicate–object extraction, joint-instance encoding,
    predicate augmentation, contrastive learning, K-Means exemplar clustering,
    LLM-assisted event naming, and event-type refinement.
  </em>
</p>

## Prerequisites

CLUE requires the following resources:

- Python and the libraries specified in the project requirements
- Stanford OpenIE for predicate–object extraction
- TextEE for event-data preprocessing
- An event extraction dataset, such as ACE, MAVEN, or RAMS

Download each dataset from its official source. Some datasets require licenses or proprietary access. Place the corresponding training JSON file in the `data/` directory.

## Running CLUE

Run the following scripts in order:

```bash
python extract_po.py
python create_instances.py
python create_masked_instances.py
python augment_verbs.py
python prep_filter_kmeans.py
python kmeans_exemplar.py
python centroid_selection.py
python ename.py
python edef.py
python merge_names.py
python merge_clus.py
```

### Dataset-specific prompts

Before running the following scripts, modify their prompt templates for the target dataset:

- `ename.py`: generates event-type names for the induced clusters.
- `edef.py`: generates fine-grained definitions for the induced event types.

The prompt examples and ontology terminology should correspond to the dataset being processed.
