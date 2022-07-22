### Module file
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS

import heapq
import pandas as pd
import networkx as nx
from tqdm.notebook import tqdm

import torch
import torch_geometric
from torch_geometric.nn.models import GNNExplainer

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ranking(model, dataset, predictions, probabilities, disease_Id, n_positive=20, explanation_nodes_ratio=1, masks_for_seed=10, G=None):

    # Predicted P(ositive) genes in the test mask
    # dictionaries with {gene: prob}
    test_P      = {}
    overall_LP  = {}

    x = dataset.x

    test_mask   = dataset.test_mask
    edge_index  = dataset.edge_index

    test_preds = predictions[test_mask]
    test_probs = probabilities[test_mask]
    test_nodes = []

    # list of positive genes indexes in test_mask
    top_k_test_P_idx = []

    # dict to store for each index the number of nodes (pos 0)
    # and edges (pos 1) in the subgraph
    subg_numnodes_d = {}

    # Dictionary in the form {seed_gene_name: [LPs in the subgraph, LPs scores from GNNExplainer]}
    candidates = {}

    # Dictionary in the from {LP: [#subgraphs it is present in, sum of GNNExplainer's scores in the different subgraphs]}
    ranking = {}

    graph_path = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'

    if G == None:
        print('[+] Reading graph...', end='')
        G = nx.read_gml(graph_path)
        print('ok')

    i = 0
    for node in G.nodes:
        if test_mask[i]:
            test_nodes.append(node)
        if predictions[i] == 1 and node not in overall_LP:
            overall_LP[node] = probabilities[i][1].item() # take probability of class 1 (LP)
        i += 1

    i = 0
    for node in test_nodes:
        if test_preds[i] == 0 and node not in test_P: #P
            test_P[node] = test_probs[i][0].item() # take probability of class 0 (p)
        i += 1

    # print('[i] # of predicted positive genes in test mask:', len(test_P))
    # print('[i] # of predicted overall likely positive genes:', len(overall_LP))

    top_k_test_P = heapq.nlargest(n_positive, test_P, key=test_P.get)

    for node in top_k_test_P:
        i = 0
        for n in G.nodes:
            if node == n:
                top_k_test_P_idx.append(i)
                break
            i += 1

    for i in top_k_test_P_idx:
        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(i, 1, edge_index)
        if i not in subg_numnodes_d:
            subg_numnodes_d[i] = [len(subg_nodes), subg_edge_index.shape[1]]

    for i in tqdm(range(len(top_k_test_P_idx))):
        nodes_names = list(G.nodes)

        idx = top_k_test_P_idx[i]
        idx_name = nodes_names[idx]
        candidates[idx_name] = {}

        mean_mask = torch.zeros(edge_index.shape[1]).to('cpu')

        for i in range(masks_for_seed):
            explainer = GNNExplainer(model, epochs=200, return_type='log_prob', num_hops=1, log=False)
            node_feat_mask, edge_mask = explainer.explain_node(idx, x, edge_index)
            mean_mask += edge_mask.to('cpu')

        mean_mask = torch.div(mean_mask, masks_for_seed)

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))

        values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1]) #take ordered list of all edges

        seen_genes = set()

        for i in range(len(indices)):
            src     = edge_index[0][indices[i]]
            trgt    = edge_index[1][indices[i]]

            src_name    = nodes_names[src]
            trgt_name   = nodes_names[trgt]

            src_pred    = predictions[src]
            trgt_pred   = predictions[trgt]

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != idx_name:
                seen_genes.add(src_name)
            if trgt_name != idx_name:
                seen_genes.add(trgt_name)

            if src_pred == 1: # LP
                if src_name not in candidates[idx_name]:
                    candidates[idx_name][src_name] = values[i]
                else:
                    candidates[idx_name][src_name] += values[i]

            if trgt_pred == 1: # LP
                if trgt_name not in candidates[idx_name]:
                    candidates[idx_name][trgt_name] = values[i]
                else:
                    candidates[idx_name][trgt_name] += values[i]
            
            # when the seen geens set reaches the num_nodes threshold
            # break the loop
            if len(seen_genes) >= num_nodes:
                break

    for seed in candidates:
        for candidate in candidates[seed]:
            if candidate not in ranking:
                ranking[candidate] = [1, candidates[seed][candidate].item()]
            else:
                ranking[candidate][0] += 1
                ranking[candidate][1] += candidates[seed][candidate].item()
    
    sorted_ranking  = sorted(ranking, key=lambda x: (ranking[x][0], ranking[x][1]), reverse=True)

    return sorted_ranking

def validate_with_extended_dataset(top_k, disease_Id, save_ranking_to_file=True):
    genes_in_extended       = []
    genes_not_in_extended   = []

    extended_genes          = pd.read_csv('Datasets/all_gene_disease_associations.tsv', sep='\t')
    extended_genes          = extended_genes[extended_genes['diseaseId'] == disease_Id]
    extended_genes_names    = set(extended_genes['geneSymbol'].tolist())

    fout = None
    if save_ranking_to_file:
        fout = open(PATH_TO_RANKINGS + disease_Id + '_20_Positive_Ranking.txt', 'w')

    for gene in top_k:
        if gene in extended_genes_names:
            genes_in_extended.append(gene)
        else:
            genes_not_in_extended.append(gene)

        if fout != None:
            fout.write(gene + '\n')

    if fout != None:
        fout.close()

    precision = len(genes_in_extended)
    # print('# of genes found in the extended dataset for disease', disease_Id, ':', precision)

    return precision