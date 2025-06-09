Butterfly Shape Evolution with BFFG
---

This project implements a probabilistic model for analyzing butterfly shape evolution
using the Backward Filtering Forward Guiding (BFFG) algorithm. The model is based on
stochastic differential equations (SDEs) and message passing over phylogenetic trees.

Project Structure
---
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

SDE.py                # Custom module for simulating SDE-based motion
ABFFG.py              # Core implementation of the BFFG algorithm

BFFG_male.ipynb       # Notebook for running BFFG on the full male dataset
BFFG_sub1.ipynb       # Notebook for running BFFG on subtree1 of the male dataset

Dependencies
------------
This project requires the following Python libraries:

- jax
- hyperiax
- pandas
- numpy
- matplotlib

Install all dependencies with:

    pip install jax hyperiax pandas numpy matplotlib

How to Use
----------
- To run the full model on the male dataset, open and run: BFFG_male.ipynb
- To test a smaller version of the pipeline (useful for debugging), use: BFFG_sub1.ipynb
  which uses the subtree defined in tree/malesub1.txt.

Notes
-----
- The shape data consists of 2D landmark coordinates per specimen.
- The phylogenetic tree is required to structure the inference using BFFG.
- The subtree experiments are useful for testing due to lower computational cost.