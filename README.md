<p align="center">
  <img src="Docs/XGDAG_logo.svg" alt="XGDAG logo" width=45%>
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 

# XGDAG
 
This is the official repository for [**XGDAG: eXplainable Gene-Disease Associations via Graph Neural Networks**](https://doi.org/10.1093/bioinformatics/btad482).

The repository contains scripts and notebook to run the code and reproduce the experiments in the paper.

# Data

The data generated and provided in this repository are based on PPI data from [BioGRID](https://thebiogrid.org/) and Gene-Disease Associations from [DisGeNET](https://www.disgenet.org/). The original data can be dowloaded from the related websites. Part of the analysis relies on the set of all disease associations from DisGeNET. Given the size of this file, it needs to be manually downloaded from [here](https://drive.google.com/file/d/12cyI6ds0mKQI9mcRgaf0_9v8KDZHWpQR/view?usp=sharing) and placed in the ```Datasets``` folder.

Using the aformentioned data we built graphs available for use in the ```Graphs``` folder. The script ```CreateGraph.py``` was used for this purpose.

# Prerequisites

XGDAG relies on PyTorch 1.12 and PyTorch Geometric 2.1.

In the ```CondaEnvs``` folder we provide several Conda environment configurations compatible with XGDAG.

# Train the model (optional)

Pretrained models for the diseases analized in the paper are available in the ```Models``` folder. However, it is possible to train a custom GraphSAGE model by running:

```bash
 python TrainerScript.py
```

The module contains parameters that can be adjusted at will.

# Explanation phase

To run XGDAG and additional explainers (SubgraphX, GraphSVX, and GNNExplainer):

```bash
 python ComputeRankingScript.py
```

The script will use the explainers to explain the models for the specified diseases. Gene rankings will be saved in the ```Rankings``` folder. The latter containts also rankings from additional methods (see paper) used for comparison.

# Analysis and comparison

After having computed the rankings (or using the precompued ones), it is possible to use the provided notebooks to analyze the results and generate plots like the ones shown in the paper. The notebooks ```comparison_plots_disgenet.ipynb``` and ```comparison_plots_omim.ipynb``` analyze the results on DisGeNET and OMIM+PheI datasets, respectively. Finally, ```comparison_plots_omim_vs_disgenet.ipynb``` provides a comparison of the results on both datasets.

# Citation

If you use our work, please cite our paper 😄

Andrea Mastropietro, Gianluca De Carlo, Aris Anagnostopoulos, XGDAG: explainable gene–disease associations via graph neural networks, Bioinformatics, Volume 39, Issue 8, August 2023, btad482, [https://doi.org/10.1093/bioinformatics/btad482](https://doi.org/10.1093/bioinformatics/btad482)

Special thanks to [Simone Fiacco](https://www.linkedin.com/in/simone-fiacco-27bb5a25a/) for creating the XGDAG logo.

## Contacts

For any further question, do not hesitate to drop an email [here](mailto:decarlo@diag.uniroma1.it) or [here](mailto:mastropietro@diag.uniroma1.it).
