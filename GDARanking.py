### Module file
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
import sys

import heapq
import pandas as pd
import networkx as nx
import numpy as np
import random
from tqdm.notebook import tqdm

import torch
import torch_geometric
from torch_geometric.nn.models import GNNExplainer

sys.path.append('C:/Repositories/GraphSVX')
from src.explainers import GraphSVX
from src.data import prepare_data
from src.train import evaluate, test

#reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True) #Mmmmmh, not sure if we want it

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###new methods

def predict_candidate_genes(model, dataset, predictions, disease_Id, explainability_method, explanation_nodes_ratio=1, masks_for_seed=10, num_hops = 1, G=None):
    if explainability_method.lower() == "gnnexplainer":
        return predict_candidate_genes_gnn_explainer(model, dataset, predictions, disease_Id, explanation_nodes_ratio, masks_for_seed, num_hops, G)
    elif explainability_method.lower() == "graphsvx":
        return predict_candidate_genes_graphsvx(model, dataset, predictions, disease_Id, explanation_nodes_ratio, masks_for_seed, num_hops, G)
    # elif explainability_method.lower() == "subgraphx":
    #     return predict_candidate_genes_subgraphx(model, dataset, predictions, disease_Id, explanation_nodes_ratio, masks_for_seed, num_hops, G)
    # elif explainability_method.lower() == "edgeshaper":
    #     return predict_candidate_genes_edgeshaper(model, dataset, predictions, disease_Id, explanation_nodes_ratio, masks_for_seed,num_hops, G)
    else:
        print("Invalid explainability method - not implemented.")
        return None

def predict_candidate_genes_gnn_explainer(model, dataset, predictions, disease_Id, explanation_nodes_ratio=1, masks_for_seed=10, G=None):
    x           = dataset.x
    labels      = dataset.y
    edge_index  = dataset.edge_index

    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    nodes_names = list(G.nodes)

    # Take all positive genes
    graph_path = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'
    if G == None:
        print('[+] Reading graph...', end='')
        G = nx.read_gml(graph_path)
        print('ok')
    
    i = 0
    for node in G:
        if labels[i] == 0:
            nodes_with_idxs[node] = i
        i += 1
    
    print('[+]', len(nodes_with_idxs), 'positive nodes found in the graph')

    # Get the subgraphs of every positive nodes
    for node in nodes_with_idxs:
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    for node in tqdm(nodes_with_idxs):
        idx = nodes_with_idxs[node]

        candidates[node] = {}

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
            if src_name != node:
                seen_genes.add(src_name)
            if trgt_name != node:
                seen_genes.add(trgt_name)

            if src_pred == 1: # LP
                if src_name not in candidates[node]:
                    candidates[node][src_name] = values[i]
                else:
                    candidates[node][src_name] += values[i]

            if trgt_pred == 1: # LP
                if trgt_name not in candidates[node]:
                    candidates[node][trgt_name] = values[i]
                else:
                    candidates[node][trgt_name] += values[i]
            
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

def predict_candidate_genes_gnn_explainer_only(model, dataset, predictions, disease_Id, explanation_nodes_ratio=1, masks_for_seed=10, G=None):
    x           = dataset.x
    labels      = dataset.y
    edge_index  = dataset.edge_index

    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    nodes_names = list(G.nodes)

    # Take all positive genes
    graph_path = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'
    if G == None:
        print('[+] Reading graph...', end='')
        G = nx.read_gml(graph_path)
        print('ok')
    
    i = 0
    for node in G:
        if labels[i] == 0:
            nodes_with_idxs[node] = i
        i += 1
    
    print('[+]', len(nodes_with_idxs), 'positive nodes found in the graph')

    # Get the subgraphs of every positive nodes
    for node in nodes_with_idxs:
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    for node in tqdm(nodes_with_idxs):
        idx = nodes_with_idxs[node]

        candidates[node] = {}

        mean_mask = torch.zeros(edge_index.shape[1]).to('cpu')

        for i in range(masks_for_seed):
            explainer = GNNExplainer(model, epochs=200, return_type='log_prob', num_hops=1, log=False)
            node_feat_mask, edge_mask = explainer.explain_node(idx, x, edge_index)
            mean_mask += edge_mask.to('cpu')

        mean_mask = torch.div(mean_mask, masks_for_seed)

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))

        # values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1]) #take ordered list of all edges
        mean_mask = torch.div(mean_mask, masks_for_seed)

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))

        threshold = torch.mean(mean_mask) #to discuss when an edge in important or not
        hard_mean_mask = (mean_mask >= threshold).to(torch.float) #>=

        values = mean_mask[hard_mean_mask == 1] ###check if correct!!!!
        indices = hard_mean_mask.nonzero()

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
            if src_name != node:
                seen_genes.add(src_name)
            if trgt_name != node:
                seen_genes.add(trgt_name)

            #if src_pred == 1: # LP # no needed here
            if src_name not in candidates[node]:
                candidates[node][src_name] = values[i]
            else:
                candidates[node][src_name] += values[i]

            #if trgt_pred == 1: # LP #no needed here
            if trgt_name not in candidates[node]:
                candidates[node][trgt_name] = values[i]
            else:
                candidates[node][trgt_name] += values[i]
            
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

def predict_candidate_genes_graphsvx(model, dataset, predictions, disease_Id, explanation_nodes_ratio=1, masks_for_seed=10, num_hops = 1, G=None):

    

    #graphsvx params
    num_samples = 400 #number of coaliton used to apporx shapley values
    info =  False
    multiclass = True
    fullempty = None #true to discard full and empy coalitions
    S = 1
    hv = "compute_pred"
    feat='Expectation',
    coal='SmarterSeparate'
    g='WLR_sklearn'
    regu = 0 #0 for explaining nodes, 1 for features
    vizu = False
    gpu = True

    x           = dataset.x
    labels      = dataset.y
    edge_index  = dataset.edge_index

    ranking         = {}
    candidates      = {}
    nodes_with_idxs = {}
    subg_numnodes_d = {}

    nodes_names = list(G.nodes)

    # Take all positive genes
    graph_path = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'
    if G == None:
        print('[+] Reading graph...', end='')
        G = nx.read_gml(graph_path)
        print('ok')
    
    i = 0
    for node in G:
        if labels[i] == 0:
            nodes_with_idxs[node] = i
        i += 1
    
    print('[+]', len(nodes_with_idxs), 'positive nodes found in the graph')

    # Get the subgraphs of every positive nodes
    for node in nodes_with_idxs:
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    for node in tqdm(nodes_with_idxs):
        idx = nodes_with_idxs[node]

        candidates[node] = {}
        
        explainer = GraphSVX(dataset, model, gpu)
        pred_explanations = explainer.explain([idx], num_hops,num_samples,info, multiclass,fullempty,S,hv,feat,coal,g,regu,vizu)
        current_node_explanations = pred_explanations[0] #only one eplxanation
        num_features_explanations = explainer.F #features in explanation, we only consider nodes. The order returned is [f0,..,fn,n0,...nm]. We want to set self.F to 0
        neighbors = explainer.neighbours #k_hop_subgraph_nodes
        explanations_shapley_values = current_node_explanations[0][num_features_explanations:] #explaining predicted class, it was 0 - Positive

        threshold = torch.mean(explanations_shapley_values) #mean value for threshold

        _, idxs = torch.topk(torch.from_numpy(
            np.abs(explanations_shapley_values)), neighbors.shape[0]) #num_important_nodes, with neighbors.shape[0] we take them all in order to remove them to obtain the needed sparsity

        vals = [explanations_shapley_values[idx] for idx in idxs]
        influential_nei = {}
        for idx, val in zip(idxs, vals):
            influential_nei[neighbors[idx]] = val
        nodes_and_explanations = [(item[0].item(), item[1].item()) for item in list(influential_nei.items())]
        nodes_and_explanations = {item[0]: item[1] for item in nodes_and_explanations}

        for node in nodes_and_explanations:
            if nodes_and_explanations[node] < threshold:
                del nodes_and_explanations[node]

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))

        important_nodes = list(nodes_and_explanations.keys())

        seen_genes = set()

        for i in range(len(important_nodes)):
            src = important_nodes[i]
            src_name    = nodes_names[src]
            src_pred    = predictions[src]
            

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != node:
                seen_genes.add(src_name)

            if src_pred == 1: # LP
                if src_name not in candidates[node]:
                    candidates[node][src_name] = nodes_and_explanations[src]
                else:
                    candidates[node][src_name] += nodes_and_explanations[src]
            
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


#legacy methods

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

    nodes_names = list(G.nodes)
    for i in tqdm(range(len(top_k_test_P_idx))):

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

def get_ranking_no_LP_intersection(model, dataset, predictions, probabilities, disease_Id, n_positive=20, explanation_nodes_ratio=1, masks_for_seed=5, G=None):

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
        # if predictions[i] == 1 and node not in overall_LP:
        #     overall_LP[node] = probabilities[i][1].item() # take probability of class 1 (LP)
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

    nodes_names = list(G.nodes)
    for i in tqdm(range(len(top_k_test_P_idx))):

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

        # values, indices = torch.topk(mean_mask, subg_numnodes_d[idx][1]) #take ordered list of all edges
        threshold = torch.mean(mean_mask) #to discuss when an edge in important or not
        hard_mean_mask = (mean_mask >= threshold).to(torch.float) #>=

        values = mean_mask[hard_mean_mask == 1] ###check if correct!!!!
        indices = hard_mean_mask.nonzero() ###check if correct!!!!

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

            #if src_pred == 1: # LP #we no longer check for LPs
            if src_name not in candidates[idx_name]:
                candidates[idx_name][src_name] = values[i]
            else:
                candidates[idx_name][src_name] += values[i]

            #if trgt_pred == 1: # LP
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

def validate_with_extended_dataset_no_LP(top_k, disease_Id, save_ranking_to_file=True):
    genes_in_extended       = []
    genes_not_in_extended   = []

    extended_genes          = pd.read_csv('Datasets/all_gene_disease_associations.tsv', sep='\t')
    extended_genes          = extended_genes[extended_genes['diseaseId'] == disease_Id]
    extended_genes_names    = set(extended_genes['geneSymbol'].tolist())

    fout = None
    if save_ranking_to_file:
        fout = open(PATH_TO_RANKINGS + disease_Id + '_no_LP_Ranking.txt', 'w')

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