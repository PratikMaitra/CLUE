### CLUE ####


### Pre-requisites ###

Python libraries like OpenIE and TextEE are needed to preprocess the data
The event datasets like ACE, MAVEN and RAMS have to downloaded from their source(some have proprietary access)
Put the train json file in the data folder.

###########

Run following scripts in order:

1. extract_po.py
2. create_instances.py
3. create_masked_instances.py
4. augment_verbs.py
5. prep_filter_kmeans.py
6. kmeans_exemplar.py
7. centroid_selection.py
8. ename.py
9. edef.py
10. merge_names.py
11. merge_clus.py


###################

The prompt for ename and edef have to be modified as per dataset.

