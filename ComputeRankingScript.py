from GNNTrain import predict_from_saved_model
from CreateDatasetv2 import get_dataset_from_graph
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
from GDARanking import predict_candidate_genes
import CreateDatasetv2_binary_diamond, CreateDatasetv2_binary

import os
import sys
from tqdm import tqdm
from time import perf_counter

disgenet_disease_Ids = ['C0006142',  'C0009402', 'C0023893', \
							 'C0036341',  'C0376358', 'C3714756', \
							 'C0860207',  'C0011581', 'C0005586', \
							 'C0001973']

omim_disease_Ids = ['C0006142',  'C0009402', 'C0023893']

methods = ['gnnexplainer',  'gnnexplainer_only',\
					 'graphsvx',      'graphsvx_only',    \
					 'subgraphx',     'subgraphx_only']

datasets = ['disgenet', 'omim']

def check_args(args):
		if len(args) < 3:
				if len(args) == 2 and (args[1] == '-h' or args[1] == '--help'):
						print('\n\n[Usage]: python ComputeRankingScript.py disease_id method num_cores dataset\n')
						print('- Available disease_id:\n', disgenet_disease_Ids, '\nType all to compute the ranking for all the available diseases\n')
						print('- Available methods:\n', methods, '\nType all to compute the ranking for all the available methods\n')
						print('- num_cores: type the number of cores to use to parallelize the explainability phase\n')
						print('- Available datasets:', datasets, '\nType all to run experiments and both the datasets\nWhen omim dataset is selected, only', omim_disease_Ids, 'disease_ids are available\n')
						print('=== Example runs ===')
						print('Ex1: python ComputeRankingScript.py C0006142 gnnexplainer 1 disgenet')
						print('Ex2: python ComputeRankingScript.py all all 1 all\n')
				else:
						print('\n\n[ERR] Wrong input parameters: use -h or --help to print the usage\n\n')
				return -1

		disease_Id = args[1]
		METHOD = args[2]
		num_cpus = int(args[3])
		dataset = args[4].lower()

		if disease_Id not in disgenet_disease_Ids and disease_Id.lower() != 'all':
				print('\n[ERR] Disease ID', disease_Id, 'not present in the database\n')
				return -1

		elif METHOD not in methods and METHOD.lower() != 'all':
				print('\n[ERR] Method', METHOD, 'not available\n')
				return -1
		
		elif num_cpus < 1:
				print('\n[ERR]', num_cpus,'is an invalid number of cores\n')
				return -1
		
		elif dataset != 'DisGeNET' or dataset != 'OMIM':
			print('\n[ERR]', dataset,'is an invalid dataset name\n')
			return -1
		
		return disease_Id, METHOD, num_cpus, dataset

def ranking(disease_Id, METHOD, num_cpus, filename, modality='multiclass'):

		model_name  = 'GraphSAGE_' + disease_Id + '_new_rankings_40000_0_0005'
		graph_path  = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'
		classes     = ['P', 'LP', 'WN', 'LN', 'RN']
		from_diamond = False
		if modality == 'binary':
				model_name += '_binary'
				classes = ['P', 'U']
				dataset, G = CreateDatasetv2_binary.get_dataset_from_graph(graph_path, disease_Id, quartile=False)
		else:
				dataset, G = get_dataset_from_graph(graph_path, disease_Id, quartile=False, from_diamond=from_diamond)

		model_name += '_40000_0_0005'

		preds, probs, model = predict_from_saved_model(model_name, dataset, classes, save_to_file=False)

		ranking = predict_candidate_genes(model,
																		dataset,
																		preds,
																		explainability_method=METHOD,
																		disease_Id=disease_Id,
																		explanation_nodes_ratio=1,
																		masks_for_seed=10,
																		num_hops=1,
																		G=G,
																		num_pos="all",
																		num_workers=num_cpus)

		# print('[+] Saving ranking to file', filename, end='...')

		with open(filename, 'w') as f:
				for line in ranking:
						f.write(line + '\n')

		print('ok')

def sanitized_input(prompt, accepted_values):
		res = input(prompt).strip().lower()
		if res not in accepted_values:
				return sanitized_input(prompt, accepted_values)
		return res

if __name__ == '__main__':
		t_start = perf_counter()

		args = check_args(sys.argv)

		if args == -1:
				sys.exit()
		
		disease_Id	= args[0]
		METHOD			= args[1]
		num_cpus		= args[2]
		dataset			= args[3]

		# if disease_Id != 'all':
		# 		disease_Ids = [disease_Id]
		
		if METHOD != 'all':
				methods = [METHOD]

		if dataset != 'all':
			datasets = [dataset]
		
		if disease_Id != 'all' and dataset != 'digenet' and (disease_Id not in omim_disease_Ids):
			print('[ERR] Disease', disease_Id, 'is not available for the OMIM dataset.\nPlease check the help page for the list of available diseases for the different datasets\n')
			sys.exit(-1)

		for dataset in tqdm(datasets, desc='dataset'):
			diseases = [disease_Id]
			if disease_Id == 'all':
				diseases = disgenet_disease_Ids if dataset == 'disgenet' else omim_disease_Ids
			# print('\n[i] Computing the ranking for', diseases, '(', len(diseases), ')', 'disease(s) on dataset', dataset)

			for disease in tqdm(diseases, desc='disease'):
					for METHOD in tqdm(methods, desc='method'):
							# print('\t[+] Starting', disease, 'with method', METHOD, 'on dataset', dataset)

							filename = PATH_TO_RANKINGS + disease + '_Phat_and_P_' + dataset + '_'

							modality = 'multiclass'
							if '_only' in METHOD:
									modality = 'binary'
							
							if modality == 'multiclass':
									filename += 'xgdag_' + METHOD.lower() + '.txt'
							else:
									filename += METHOD.lower().replace("_only", "") + '.txt'

							res = ''
							if os.path.exists(filename):
									res = sanitized_input('[+] A raking for disease ' + disease + \
											' has already been computed with ' + METHOD + \
											'. Do you want to overwrite the old ranking? (y|n) ', ['y', 'n'])
							if res == 'n':
									print('[i] Skipping disease', disease, 'with method', METHOD)
									continue
							else:
									# Compute the ranking
									ranking(disease, METHOD, num_cpus, filename, modality)

		t_end = perf_counter()
		print('[i] Elapsed time:', round(t_end - t_start, 3))