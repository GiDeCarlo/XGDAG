### Script file. Gianluca De Carlo & Andrea Mastropietro © All rights reserved.

### This script creates a graph from a PPI network and adds NeDBIT features to the nodes.
### The graph is saved in the GML format.
 
import sys

import numpy as np
import pandas as pd
import networkx as nx

import yaml
from tqdm.auto import tqdm

from sklearn.preprocessing import RobustScaler

def create_graph_from_PPI(path_to_PPI, from_diamond=False):
	
	# Skip if using the diamond dataset 
	if not from_diamond:
		print('[+] Reading PPI...', end='')
		biogrid = pd.read_csv(path_to_PPI, sep='\t', low_memory=False)
		# Filtering non-human proteins
		biogrid = biogrid[(biogrid['Organism ID Interactor A'] == 9606) & (biogrid['Organism ID Interactor B'] == 9606)]
		print('ok')
	
	print('[+] Creating the graph...', end='')
	G = nx.Graph()

	if from_diamond:
		with open(path_to_PPI, 'r') as ppi:
			for line in ppi.readlines():
					line = line.strip().split(' ')
					p1 = line[0]
					p2 = line[1]
					G.add_edge(p1, p2)
					
	elif not from_diamond:
		for index, row in biogrid.iterrows():
				p1 = row['Official Symbol Interactor A'].replace('-', '_').replace('.', '_')
				p2 = row['Official Symbol Interactor B'].replace('-', '_').replace('.', '_')
				G.add_edge(p1, p2)

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

	return G

def add_nedbit_features(G, feature_path, graph_save_path, disease_id, scale=False):
	print('[+] Adding NeDBIT features...', end='')

	feature_path_disease = feature_path + disease_id + '_features.csv'
	try:
		nedbit_features = pd.read_csv(feature_path_disease)
	except FileNotFoundError:
		print('Error: file ' + feature_path_disease + ' not found. Exiting...')
		sys.exit(-1) 

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

	print('[+] Saving graph to path:', graph_save_path)
	nx.write_gml(G, graph_save_path)

	return graph_save_path

if __name__ == '__main__':
	args = None

	with open("parameters.yml") as paramFile:  
		args = yaml.load(paramFile, Loader=yaml.FullLoader)

	PATH_TO_PPI  = args["create_graph_from_PPI"]["PATH_TO_PPI"]
	DISEASE_IDs = args["create_graph_from_PPI"]["DISEASE_IDs"]
	SCALE_FEATURES = args["create_graph_from_PPI"]["SCALE_FEATURES"]
	FEATURE_PATH = args["create_graph_from_PPI"]["FEATURE_PATH"]
	FROM_DIAMOND = args["create_graph_from_PPI"]["FROM_DIAMOND"]
	GRAPH_SAVE_PATH = args["create_graph_from_PPI"]["GRAPH_SAVE_PATH"]

	G = create_graph_from_PPI(PATH_TO_PPI, FROM_DIAMOND)

	for disease_id in tqdm(DISEASE_IDs):
		graph_name = 'graph_nedbit_'

		if FROM_DIAMOND:
			graph_name = 'graph_diamond_nedbit_'
		graph_name += disease_id + '.gml'

		graph_save_path = GRAPH_SAVE_PATH + graph_name
		
		add_nedbit_features(G.copy(), FEATURE_PATH, graph_save_path, disease_id, SCALE_FEATURES)