### Module file
import torch
import numpy as np
import pandas as pd
import networkx as nx
from time import perf_counter

from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset

from sklearn.model_selection import train_test_split

from Paths import PATH_TO_DATASETS

torch.manual_seed(42)

class MyDataset(InMemoryDataset):
  def __init__(self, G, labels, attributes, num_classes=2):
    super(MyDataset, self).__init__('.', None, None, None)

    # import data from the networkx graph with the attributes of the nodes
    data = from_networkx(G, attributes)
      
    y = torch.from_numpy(labels).type(torch.long)

    data.x = data.x.float()
    data.y = y.clone().detach()
    data.num_classes = num_classes

    # Using train_test_split function from sklearn to stratify train/test/val sets
    indices = range(G.number_of_nodes())
    # Stratified split of train/test/val sets. Returned indices are used to create the masks
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(data.x, data.y, indices, test_size=0.3, stratify=labels, random_state=42)
    # To create validation set, test set is splitted in half
    X_test, X_val, y_test, y_val, test_idx, val_idx = train_test_split(X_test, y_test, test_idx, test_size=0.5, stratify=y_test, random_state=42)

    n_nodes = G.number_of_nodes()
    train_mask  = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask   = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask    = torch.zeros(n_nodes, dtype=torch.bool)
    
    for idx in train_idx:
      train_mask[idx] = True

    for idx in test_idx:
      test_mask[idx] = True
    
    for idx in val_idx:
      val_mask[idx] = True

    data['train_mask']  = train_mask
    data['test_mask']   = test_mask
    data['val_mask']    = val_mask

    self.data, self.slices = self.collate([data])

# Given the path to the gml file of the graph and the disease id,
# an instance of MyDataset class is returned. MyDataset is a class
# to create a custom dataset for training a pytorch model

def get_dataset_from_graph(path_to_graph, disease_id, verbose=True):
    t_start = perf_counter()

    if verbose: print('[+] Reading graph...', end='')
    G = nx.read_gml(path_to_graph)
    if verbose: print('ok')

    if verbose: print('[+] Creating dataset...', end='')
    path_to_seed_genes = PATH_TO_DATASETS + disease_id + '_seed_genes.txt'

    seed_genes          = pd.read_csv(path_to_seed_genes, header=None, sep=' ')
    seed_genes.columns  = ["name", "GDA Score"]
    seeds_list          = seed_genes["name"].values.tolist()
    
    nedbit_scores = pd.read_csv(PATH_TO_DATASETS + disease_id + '_features_Score.csv')

    # Remove seed genes
    nedbit_scores_not_seed = nedbit_scores[~nedbit_scores['name'].isin(seeds_list)]

    # Sort scores for quartile division
    nedbit_scores_not_seed = nedbit_scores_not_seed.sort_values(by = "out", ascending = False)
    pseudo_labels = pd.qcut(x = nedbit_scores_not_seed["out"], q = 4, labels = ["RN", "LN", "WN", "LP"])

    nedbit_scores_not_seed['label'] = pseudo_labels

    nedbit_scores_seed = nedbit_scores[nedbit_scores['name'].isin(seeds_list)]
    nedbit_scores_seed = nedbit_scores_seed.assign(label = 'P')

    # Convert dataframe to dict for searching nodes and their labels
    not_seed_labels = dict(zip(nedbit_scores_not_seed['name'], nedbit_scores_not_seed['label']))
    seed_labels     = dict(zip(nedbit_scores_seed['name'], nedbit_scores_seed['label']))

    labels_dict = {'P':0, 'LP': 1, 'WN': 2, 'LN': 3, 'RN': 4}
    labels = []

    for node in G:
        if node in not_seed_labels:
            labels.append(labels_dict[not_seed_labels[node]])
        else:
            labels.append(labels_dict[seed_labels[node]])

    labels = np.asarray(labels)

    attributes = ['degree', 'ring', 'NetRank', 'NetShort', 'HeatDiff', 'InfoDiff']

    dataset_with_nedbit = MyDataset(G, labels, attributes, num_classes=5)
    if verbose: print('ok')

    data_with_nedbit = dataset_with_nedbit[0]
    
    t_end = perf_counter()
    if verbose: print('[i] Elapsed time:', round(t_end - t_start, 3))

    return data_with_nedbit, G