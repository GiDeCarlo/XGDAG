### Module file
import numpy as np
import pandas as pd
import networkx as nx
from time import perf_counter

from sklearn.preprocessing import RobustScaler

from Paths import PATH_TO_DATASETS, PATH_TO_GRAPHS

def create_graph_from_PPI(path_to_PPI, disease_id, graph_name, scale=False):
    t_start = perf_counter()

    # print('[+] Reading PPI...', end='')
    # biogrid = pd.read_csv(path_to_PPI, sep=' ', low_memory=False)

    # Filtering non-human proteins
    # biogrid = biogrid[(biogrid['Organism ID Interactor A'] == 9606) & (biogrid['Organism ID Interactor B'] == 9606)]
    # print('ok')
    
    print('[+] Creating the graph...', end='')
    G = nx.Graph()

    with open(path_to_PPI, 'r') as ppi:
        for line in ppi.readlines():
            line = line.strip().split(' ')
            p1 = line[0]
            p2 = line[1]
            G.add_edge(p1, p2)

    # for index, row in biogrid.iterrows():
    #     p1 = row['Official Symbol Interactor A'].replace('-', '_').replace('.', '_')
    #     p2 = row['Official Symbol Interactor B'].replace('-', '_').replace('.', '_')

    #     G.add_edge(p1, p2)
    print('ok')

    print('\t[+] Added', len(list(G.nodes)), 'nodes')
    print('\t[+] Added', len(list(G.edges)), 'edges')

    # Remove self loops
    print('[+] Removing self loops...', end='')
    G.remove_edges_from(nx.selfloop_edges(G))
    print('ok')

    print('\t[+]', len(list(G.nodes)), 'nodes')
    print('\t[+]', len(list(G.edges)), 'edges')

    # Let's tale only the largest connected component
    print('[+] Taking the LCC...', end='')
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    print('ok')

    print('\t[+]', len(list(G.nodes)), 'nodes')
    print('\t[+]', len(list(G.edges)), 'edges')

    print('[+] Adding NeDBIT features...', end='')
    nedbit_features = pd.read_csv(PATH_TO_DATASETS + 'Diamond_dataset/' + disease_id + '_nedbit_features', keep_default_na=False)

    degree      = dict(zip(nedbit_features['name'], nedbit_features['degree']))
    ring        = dict(zip(nedbit_features['name'], nedbit_features['ring']))
    NetRank     = dict(zip(nedbit_features['name'], nedbit_features['NetRank']))
    NetShort    = dict(zip(nedbit_features['name'], nedbit_features['NetShort']))
    HeatDiff    = dict(zip(nedbit_features['name'], nedbit_features['HeatDiff']))
    InfoDiff    = dict(zip(nedbit_features['name'], nedbit_features['InfoDiff']))

    for node in G:
        G.nodes[node]['degree']    = degree[node]
        G.nodes[node]['ring']      = ring[node]
        G.nodes[node]['NetRank']   = NetRank[node]
        G.nodes[node]['NetShort']  = NetShort[node]
        G.nodes[node]['HeatDiff']  = HeatDiff[node]
        G.nodes[node]['InfoDiff']  = InfoDiff[node]
    print('ok')

    if scale:
        print('[+] Normalizing NeDBIT features...', end='')
        degree      = []
        ring        = []
        NetRank     = []
        NetShort    = []
        HeatDiff    = []
        InfoDiff    = []

        for node in G:
            degree.append(G.nodes[node]['degree'])
            ring.append(G.nodes[node]['ring'])
            NetRank.append(G.nodes[node]['NetRank'])
            NetShort.append(G.nodes[node]['NetShort'])
            HeatDiff.append(G.nodes[node]['HeatDiff'])
            InfoDiff.append(G.nodes[node]['InfoDiff'])

        features = [degree, ring, NetRank, NetShort, HeatDiff, InfoDiff]

        transformer = RobustScaler().fit(np.array(features))
        features = transformer.transform(np.array(features))

        i = 0
        for node in G:
            G.nodes[node]['degree']    = features[0][i]
            G.nodes[node]['ring']      = features[1][i]
            G.nodes[node]['NetRank']   = features[2][i]
            G.nodes[node]['NetShort']  = features[3][i]
            G.nodes[node]['HeatDiff']  = features[4][i]
            G.nodes[node]['InfoDiff']  = features[5][i]
            i += 1
        print('ok')
    
    graph_path = PATH_TO_GRAPHS + graph_name + '.gml'
    print('[+] Saving graph to path:', graph_path)
    nx.write_gml(G, graph_path)

    t_end = perf_counter()
    print('[i] Elapsed time:', round(t_end - t_start, 3))

    return graph_path
