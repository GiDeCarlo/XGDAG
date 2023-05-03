![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 

# XGDAG
 
This is the official repository for **XGDAG: eXplainable Gene-Disease Associations via Graph Neural Networks**.

The repository contains scripts and notebook to run the code and reproduce the experiments in the paper.

# Data

The data generated and provided in this repository are based on PPI data from [BioGRID](https://thebiogrid.org/) and Gene-Disease Associations from [DisGeNET](https://www.disgenet.org/). The original data can be dowloaded from the related websites.

Using the aformentioned data we built graphs available for use in the ```Graphs``` folder. The script ```CreateGraph.py``` was used for this purpose.

# Train the model (optional)

We already provide pretrained models for the diseases analized in the paper in the ```Models``` folder. However, it is possible to train a custom GraphSAGE model by running:

```bash
 python TrainerScript.py
```

The module contains parameters that can be adjusted at will.

# Explanation phase

To run XGDAG and additional explainers (SubgraphX, GraphSVX, and GNNExplainer):

```bash
 python ComputeRankingScript.py
```

The gene rankings will be saved in the ```Rankings``` folder. The latter containts also rankings from additional methods (see paper).

# Explanation analyses and comparison

Additionally, notebooks to reproduce the analyses shown in the paper are available (```comparison_plots.ipynb``` and ```comparison_plots_omim_disgenet.ipynb```). Those notebooks load precomputed explanations and generate plots.
