BUTTERFLY SHAPE EVOLUTION WITH BFFG
---

This project implements a probabilistic model for analysing butterfly shape evolution
with the Backward Filtering Forward Guiding (BFFG) algorithm. The model uses
stochastic differential equations (SDEs) and message passing over phylogenetic trees.


PROJECT STRUCTURE
---

CONTACT MICHAEL SEVERINSEN IN ORDER TO COLLECT THE DATA AND THE REAL TREE STRUCTURE, NOT JUST THE SUBSAMPLES. I HAVE ONLY INCLUDED THE SUBSAMPLES, AND THE NOTEBOOK RUN FOR SUBTREE 1. IT DID NOT COMPLETE THE RUN FOR THE BIG MALE DATA SET.

data/
    female.csv         # Landmark coordinates for female butterflies
    male.csv           # Landmark coordinates for male butterflies
    graphium.csv       # Landmark coordinates for Graphium genus

tree/
    female_tree.txt    # Phylogenetic tree for female data
    male_tree.txt      # Phylogenetic tree for male data
    graphium_tree.txt  # Phylogenetic tree for graphium data
    malesub1.txt       # Subtree 1 from male tree
    malesub2.txt       # Subtree 2 from male tree
    malesub3.txt       # Subtree 3 from male tree

SDE.py                # Custom module for simulating SDE
ABFFG.py              # Implementation of the BFFG algorithm

BFFG_male.ipynb       # Notebook for running BFFG on the full male dataset
BFFG_sub1.ipynb       # Notebook for running BFFG on subtree1 of the male dataset


INSTALLATIONS
------------
This project requires the following Python libraries:

- jax
- hyperiax
- pandas
- numpy
- matplotlib

Install all dependencies with:

    pip install jax hyperiax pandas numpy matplotlib