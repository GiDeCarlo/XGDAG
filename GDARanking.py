### Module file
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS

import heapq
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import multiprocessing

import torch
import torch_geometric
from torch_geometric.nn.models import GNNExplainer

from GraphSVX.src.explainers import GraphSVX
# from src.data import prepare_data
# from src.train import evaluate, test

from SubgraphX import SubgraphX

#reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### New methods
### Wrapper function
def predict_candidate_genes(model, dataset, predictions, disease_Id, explainability_method,
                            explanation_nodes_ratio=1, masks_for_seed=10, num_hops=1,
                            G=None, num_pos="all", threshold=False, num_workers=1):
    
    print('[i] Device:', device)

    if explainability_method.lower() == "gnnexplainer":
        return predict_candidate_genes_gnn_explainer(model,
                                                     dataset,
                                                     predictions,
                                                     disease_Id,
                                                     explanation_nodes_ratio,
                                                     masks_for_seed = masks_for_seed,
                                                     num_pos = num_pos,
                                                     G=G
                                                    )
    
    elif explainability_method.lower() == "gnnexplainer_only":
        return predict_candidate_genes_gnn_explainer_only(model,
                                                          dataset,
                                                          predictions,
                                                          disease_Id,
                                                          explanation_nodes_ratio=explanation_nodes_ratio,
                                                          masks_for_seed=masks_for_seed,
                                                          num_pos=num_pos,
                                                          G=G
                                                          )

    elif explainability_method.lower() == "graphsvx":
        return predict_candidate_genes_graphsvx(model,
                                                dataset,
                                                predictions,
                                                disease_Id,
                                                explanation_nodes_ratio=explanation_nodes_ratio,
                                                threshold=threshold,
                                                num_hops=num_hops,
                                                num_pos=num_pos,
                                                G=G
                                                )

    elif explainability_method.lower() == "graphsvx_only":
        return predict_candidate_genes_graphsvx_only(model,
                                                     dataset,
                                                     predictions,
                                                     disease_Id,
                                                     explanation_nodes_ratio=explanation_nodes_ratio,
                                                     num_hops=num_hops,
                                                     G=G,
                                                     num_pos=num_pos,
                                                     threshold = True
                                                    )

    elif explainability_method.lower() == "subgraphx":
        return predict_candidate_genes_subgraphx(model,
                                                 dataset,
                                                 predictions,
                                                 explanation_nodes_ratio,
                                                 num_hops,
                                                 G,
                                                 num_workers=num_workers
                                                )

    elif explainability_method.lower() == "subgraphx_only":
        return predict_candidate_genes_subgraphx(model,
                                                 dataset,
                                                 predictions,
                                                 explanation_nodes_ratio,
                                                 num_hops,
                                                 G,
                                                 num_workers=num_workers,
                                                 num_classes=2
                                                ) 
    # elif explainability_method.lower() == "edgeshaper":
    #     return predict_candidate_genes_edgeshaper(model, dataset, predictions, disease_Id, explanation_nodes_ratio, masks_for_seed,num_hops, G)
    else:
        print("Invalid explainability method || not implemented.")
        return None

def predict_candidate_genes_gnn_explainer(model, dataset, predictions, disease_Id, explanation_nodes_ratio=1, masks_for_seed=10, G=None, num_pos='all'):
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

    if num_pos == "all":
        num_pos = len(nodes_with_idxs)

    # Get the subgraphs of every positive nodes
    for node in nodes_with_idxs:
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    nodes_explained = 0
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

        nodes_explained += 1
        if num_pos != len(nodes_with_idxs) and nodes_explained >= num_pos:
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

def predict_candidate_genes_gnn_explainer_only(model, dataset, predictions, disease_Id, explanation_nodes_ratio=1, masks_for_seed=10, G=None, num_pos='all'):
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

    if num_pos == "all":
        num_pos = len(nodes_with_idxs)

    # Get the subgraphs of every positive nodes
    for node in tqdm(nodes_with_idxs):
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]
    
    nodes_explained = 0
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

            if src_pred == 1: # LP # no needed here but unlabelled
                if src_name not in candidates[node]:
                    candidates[node][src_name] = values[i]
                else:
                    candidates[node][src_name] += values[i]

            if trgt_pred == 1: # LP #no needed here but unlabelled
                if trgt_name not in candidates[node]:
                    candidates[node][trgt_name] = values[i]
                else:
                    candidates[node][trgt_name] += values[i]
            
            # when the seen geens set reaches the num_nodes threshold
            # break the loop
            if len(seen_genes) >= num_nodes:
                break
        
        nodes_explained += 1
        if num_pos != len(nodes_with_idxs) and nodes_explained >= num_pos:
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

def predict_candidate_genes_graphsvx(model, dataset, predictions, disease_Id, explanation_nodes_ratio=1, num_hops=1, G=None, num_pos="all", threshold = False):
    # print(num_pos)
    #graphsvx params
    num_samples = 100 #number of coaliton used to apporx shapley values
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

    # x           = dataset.x.to('cpu')
    labels      = dataset.y.to('cpu')
    edge_index  = dataset.edge_index.to('cpu')
    

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

    if num_pos == "all":
        num_pos = len(nodes_with_idxs)

    # Get the subgraphs of every positive nodes
    
    for node in tqdm(nodes_with_idxs):
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    nodes_explained = 0
    for node in tqdm(nodes_with_idxs):

        idx = nodes_with_idxs[node]

        candidates[node] = {}
        
        explainer = GraphSVX(dataset.to("cpu"), model, gpu)
        pred_explanations = explainer.explain([idx], num_hops,num_samples,info, multiclass,fullempty,S,hv,feat,coal,g,regu,vizu)
        current_node_explanations = pred_explanations[0] #only one eplxanation
        num_features_explanations = explainer.F #features in explanation, we only consider nodes. The order returned is [f0,..,fn,n0,...nm]. We want to set self.F to 0
        neighbors = explainer.neighbours #k_hop_subgraph_nodes
        explanations_shapley_values = current_node_explanations[0][num_features_explanations:] #explaining predicted class, it was 0 - Positive

        if threshold:
            threshold = np.mean(explanations_shapley_values) #mean value for threshold

        _, idxs = torch.topk(torch.from_numpy(
            np.abs(explanations_shapley_values)), neighbors.shape[0]) #num_important_nodes, with neighbors.shape[0] we take them all in order to remove them to obtain the needed sparsity

        vals = [explanations_shapley_values[idx] for idx in idxs]
        influential_nei = {}
        for idx_n, val in zip(idxs, vals):
            influential_nei[neighbors[idx_n]] = val
            
        nodes_and_explanations = [(item[0].item(), item[1].item()) for item in list(influential_nei.items())]
        nodes_and_explanations = {item[0]: item[1] for item in nodes_and_explanations}
    
        if threshold:
            nodes_and_scores_candidates = {item[0]: item[1] for item in nodes_and_explanations.items() if item[1] >= threshold}
        else:
            nodes_and_scores_candidates = nodes_and_explanations

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))
        print(subg_numnodes_d[idx][0])
        important_nodes = list(nodes_and_scores_candidates.keys())

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
                print('break')
                break
        
        nodes_explained += 1
        if num_pos != len(nodes_with_idxs) and nodes_explained >= num_pos:
            break
        

    for seed in candidates:
        for candidate in candidates[seed]:
            if candidate not in ranking:
                ranking[candidate] = [1, candidates[seed][candidate]]
            else:
                ranking[candidate][0] += 1
                ranking[candidate][1] += candidates[seed][candidate]
    
    sorted_ranking  = sorted(ranking, key=lambda x: (ranking[x][0], ranking[x][1]), reverse=True)

    return sorted_ranking

def predict_candidate_genes_graphsvx_only(model, dataset, predictions, disease_Id, explanation_nodes_ratio=1, num_hops=1, G=None, num_pos="all", threshold = True):
    print(num_pos)
    #graphsvx params
    num_samples = 100 #number of coaliton used to apporx shapley values
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

    # x         = dataset.x.to('cpu')
    labels      = dataset.y.to('cpu')
    edge_index  = dataset.edge_index.to('cpu')
    
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

    if num_pos == "all":
        num_pos = len(nodes_with_idxs)

    # Get the subgraphs of every positive nodes
    
    for node in nodes_with_idxs:
        idx = nodes_with_idxs[node]

        subg_nodes, subg_edge_index, subg_mapping, subg_edge_mask = torch_geometric.utils.k_hop_subgraph(idx, 1, edge_index)
        if idx not in subg_numnodes_d:
            subg_numnodes_d[idx] = [len(subg_nodes), subg_edge_index.shape[1]]

    # Get explanations of all the positive genes
    nodes_explained = 0
    for node in tqdm(nodes_with_idxs):

        idx = nodes_with_idxs[node]

        candidates[node] = {}
        
        explainer = GraphSVX(dataset.to("cpu"), model, gpu)
        pred_explanations = explainer.explain([idx], num_hops,num_samples,info, multiclass,fullempty,S,hv,feat,coal,g,regu,vizu)
        current_node_explanations = pred_explanations[0] #only one eplxanation
        num_features_explanations = explainer.F #features in explanation, we only consider nodes. The order returned is [f0,..,fn,n0,...nm]. We want to set self.F to 0
        neighbors = explainer.neighbours #k_hop_subgraph_nodes
        explanations_shapley_values = current_node_explanations[0][num_features_explanations:] #explaining predicted class, it was 0 - Positive

        _, idxs = torch.topk(torch.from_numpy(
            np.abs(explanations_shapley_values)), neighbors.shape[0]) #num_important_nodes, with neighbors.shape[0] we take them all in order to remove them to obtain the needed sparsity

        vals = [explanations_shapley_values[idx] for idx in idxs]
        influential_nei = {}
        for idx_n, val in zip(idxs, vals):
            influential_nei[neighbors[idx_n]] = val
            
        nodes_and_explanations = [(item[0].item(), item[1].item()) for item in list(influential_nei.items())]
        nodes_and_explanations = {item[0]: item[1] for item in nodes_and_explanations}
    
        if threshold:
            threshold_value = np.mean(list(nodes_and_explanations.values()))
            nodes_and_scores_candidates = {item[0]: item[1] for item in nodes_and_explanations.items() if item[1] >= threshold_value}
        else:
            nodes_and_scores_candidates = nodes_and_explanations

        num_nodes = int(round(subg_numnodes_d[idx][0]*explanation_nodes_ratio))
        print(subg_numnodes_d[idx][0])
        important_nodes = list(nodes_and_scores_candidates.keys())

        seen_genes = set()

        for i in range(len(important_nodes)):
            src = important_nodes[i]
            src_name    = nodes_names[src]
            src_pred    = predictions[src]
            

            # if gene has not been seen and it is not the explained node
            # we add it to the seen genes set
            if src_name != node:
                seen_genes.add(src_name)

            if src_pred == 1: # here 1 is unlabelled. We look for candidates in the unlabelled set
                if src_name not in candidates[node]:
                    candidates[node][src_name] = nodes_and_explanations[src]
                else:
                    candidates[node][src_name] += nodes_and_explanations[src]
            
            # when the seen geens set reaches the num_nodes threshold
            # break the loop

            if len(seen_genes) >= num_nodes:
                print('break')
                break
        
        nodes_explained += 1
        if num_pos != len(nodes_with_idxs) and nodes_explained >= num_pos:
            break
        

    for seed in candidates:
        for candidate in candidates[seed]:
            if candidate not in ranking:
                ranking[candidate] = [1, candidates[seed][candidate]]
            else:
                ranking[candidate][0] += 1
                ranking[candidate][1] += candidates[seed][candidate]
    
    sorted_ranking  = sorted(ranking, key=lambda x: (ranking[x][0], ranking[x][1]), reverse=True)

    return sorted_ranking

def run_explanation(args):
    node    = args[0]
    model   = args[1]
    G       = args[2]
    predictions = args[3]
    num_hops    = args[4]
    edge_index  = args[5]
    explanation_nodes_ratio = args[6]
    x = args[7]
    num_classes = args[8]

    pid = multiprocessing.current_process().pid

    print('[', pid, '] New worker created. Explaining node', node)

    nodes_list = list(G.nodes)
    idx = nodes_list.index(node)
    prediction = predictions[idx]

    candidates = {}
    candidates[node] = {}
    # get candidates for node
    explainer = SubgraphX.SubgraphX(model,
                    num_classes=num_classes,
                    device=device,
                    num_hops=num_hops,
                    min_atoms=2,
                    explain_graph=False,
                    reward_method='nc_mc_l_shapley')
    
    original_mapping, _, _, _ = SubgraphX.k_hop_subgraph_with_default_whole_graph(
            edge_index=edge_index,
            node_idx=idx,
            num_hops=1,
        )
    
    max_nodes = int(round(len(original_mapping) * explanation_nodes_ratio))

    _, explanation_results, _ = explainer(x, edge_index, node_idx=idx, max_nodes=max_nodes)

    best_coalition = SubgraphX.find_closest_node_result_fixed_nodes(explanation_results[prediction], max_nodes=max_nodes)

    score = best_coalition['P'] # Get score computed by SubgraphX for the entire subgraph

    ori_nodes_idxs = [original_mapping[n].item() for n in best_coalition['coalition']]

    for coalition_node_idx in ori_nodes_idxs:
        if coalition_node_idx != idx:
            coalition_node_name = nodes_list[coalition_node_idx]
            
            if predictions[coalition_node_idx] == 1: # if node is LP
                candidates[node][coalition_node_name] = score
    
    print('[', pid, '] Worker on node', node, 'done')

    return candidates

def predict_candidate_genes_subgraphx(model, dataset, predictions, explanation_nodes_ratio, num_hops, G, num_workers=1, num_classes=5):
    labels = dataset.y.to('cpu')
    node_list = list(G.nodes)

    host_cpu_count = multiprocessing.cpu_count()
    if num_workers > host_cpu_count:
        print('\t[i] Passed', num_workers, 'as num_cores, but is seems that you have only', host_cpu_count,\
            'to avoid errors, num_cores is set to', host_cpu_count)
        num_workers = host_cpu_count
    
    print('[i] Using', num_workers, 'cores')

    max_degree = 20

    print('[i] Filtering seed genes with more than', max_degree, 'degree to reduce computational time of SubgraphX.')

    parameters_l = []
    # Get positive nodes
    for i in range(len(node_list)):
        node = node_list[i]
        if labels[i] == 0 and G.degree[node] < max_degree: # Degree filter
            parameters_l.append([node, model, G, predictions, num_hops, dataset.edge_index, explanation_nodes_ratio, dataset.x, num_classes])
    
    print('[i]', len(parameters_l), 'seed genes selected.')
    
    if device == 'cuda':
        multiprocessing.set_start_method('spawn', force=True)

    p = multiprocessing.Pool(num_workers)
    candidates_list = p.map(run_explanation, parameters_l)
    p.close()

    ranking = {}

    for candidates in candidates_list:
        for seed in candidates:
            for candidate in candidates[seed]:
                if candidate not in ranking:
                    ranking[candidate] = [1, candidates[seed][candidate]]
                else:
                    ranking[candidate][0] += 1
                    ranking[candidate][1] += candidates[seed][candidate]

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