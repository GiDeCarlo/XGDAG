![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 

# XGDAG
 
This is the official repository for XGDAG: eXplainable Gene-Disease Associations via Graph Neural Networks

The repository contains scripts and notebook to run the code and reproduce the experiments in the paper.

We provide pretrained GraphSAGE models that can be used for the explanation phase. In order to use such models and explain using XGDAG and additional XAI methods, simply run:

 ```bash
 python ComputeRankingScript.py
 ```

Moreover, the module ```GNNTrain``` provides facilities to train a model on custom data. We provide the implementation of the GraphSAGE model employed in the paper in ```GraphSageModel.py```. However, any custom model can be trained and explained using the XGDAG framework.

Additionally, notebooks to reproduce the analyses shown in the paper are available (```comparison_plots.ipynb``` and ```comparison_plots_omim_disgenet.ipynb```). Those notebooks load precomputed explanations and generate plots.
