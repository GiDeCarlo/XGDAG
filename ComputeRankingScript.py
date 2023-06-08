from GNNTrain import predict_from_saved_model
from CreateDatasetv2 import get_dataset_from_graph
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
from GDARanking import predict_candidate_genes
import CreateDatasetv2_binary_diamond, CreateDatasetv2_binary

import os
import sys
import numpy as np
import multiprocessing
from time import perf_counter
from sklearn.metrics import classification_report

disgenet_disease_Ids = ['C0006142',  'C0009402', 'C0023893', \
							 'C0036341',  'C0376358', 'C3714756', \
							 'C0860207',  'C0011581', 'C0005586', \
							 'C0001973']

omim_disease_Ids = ['C0006142',  'C0009402', 'C0023893']

available_methods = ['gnnexplainer', 'gnnexplainer_only',\
					 						'graphsvx', 'graphsvx_only',    \
					 						'subgraphx', 'subgraphx_only']

available_datasets = ['disgenet', 'omim']

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

	diseases = args[1].split(',')
	methods = args[2].split(',')
	num_cpus = int(args[3])
	dataset = args[4].lower()
	
	if diseases[0] != 'all':
		for disease in diseases:
			if disease not in disgenet_disease_Ids:
				print('\n[ERR] Disease ID', disease, 'not present in the database\n')
				return -1

	if methods[0] != 'all':
		for method in methods:
			if method not in available_methods:
				print('\n[ERR] Method', method, 'not available\n')
				return -1
	
	if num_cpus < 1:
			print('\n[ERR]', num_cpus,'is an invalid number of cores\n')
			return -1
	
	if dataset not in available_datasets and dataset != 'all':
		print('\n[ERR]', dataset,'is an invalid dataset name\n')
		return -1
	
	if dataset == 'omim':
		for disease in diseases:
			if disease not in omim_disease_Ids:
				print('\n[ERR]', disease,' not available for dataset', dataset, '\n')
				return -1
	
	return diseases, methods, num_cpus, dataset

def ranking(args):

	disease_Id		= args[0]
	method				= args[1]
	num_cpus			= args[2]
	filename			= args[3]
	dataset_name	= args[4]
	modality			= args[5]

	print('[+] Process', os.getpid(), 'STARTED. Configuration:', disease_Id, method, dataset_name)

	start_time = perf_counter()

	if dataset_name == 'disgenet':
		model_name  = 'GraphSAGE_' + disease_Id + '_new_rankings_'
		graph_path  = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'
	else: # omim
		model_name = 'GraphSAGE_' + disease_Id + '_diamond_'
		graph_path  = PATH_TO_GRAPHS + 'grafo_diamond_nedbit_' + disease_Id + '.gml'
	
	classes     = ['P', 'LP', 'WN', 'LN', 'RN']

	if modality == 'binary':
			model_name += 'binary_'
			classes = ['P', 'U']
			if dataset_name == 'disgenet':
				dataset, G = CreateDatasetv2_binary.get_dataset_from_graph(graph_path, disease_Id, quartile=False, verbose=False)
			else:
				dataset, G = CreateDatasetv2_binary_diamond.get_dataset_from_graph(graph_path, disease_Id, quartile=False, verbose=False)
	else:
			dataset, G = get_dataset_from_graph(graph_path, disease_Id, quartile=False, from_diamond=(dataset_name=='omim'), verbose=False)

	model_name += '40000_0_0005'

	preds, probs, model = predict_from_saved_model(model_name, dataset, classes, save_to_file=False, plot_results=False)

	print(classification_report(dataset.y.to('cpu'), preds.to('cpu')))
	
	print('count 0 in labels', (dataset.y.cpu().numpy() == 0).sum())
	print('count 0 in preds', (preds.cpu().numpy() == 0).sum())
  
  # Count the number of elements that are equal to the specific value
	count = np.count_nonzero((dataset.y.cpu().numpy() == 0) & (preds.cpu().numpy() == 0))
	print("P elements predicted as P:", count)

	print('count 1 in labels', (dataset.y.cpu().numpy() == 1).sum())
	print('count 1 in preds', (preds.cpu().numpy() == 1).sum())
  
  # Find the common elements between the two arrays
	count = np.count_nonzero((dataset.y.cpu().numpy() == 1) & (preds.cpu().numpy() == 1))
	print("LP (or U) elements predicted as LP (or U):", count)
  
	count = np.count_nonzero((dataset.y.cpu().numpy() == 1) & (preds.cpu().numpy() == 0))
	print("LP (or U) elements predicted as P:", count)

	ranking = predict_candidate_genes(model,
																	dataset,
																	preds,
																	explainability_method=method,
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

	end_time = round(perf_counter()-start_time, 3)

	print('[+] Process', os.getpid(), 'FINISHED. Configuration:', disease_Id, method, dataset_name,' - ', end_time, 'seconds')

def sanitized_input(prompt, accepted_values):
		res = input(prompt).strip().lower()
		if res not in accepted_values:
				return sanitized_input(prompt, accepted_values)
		return res

if __name__ == '__main__':
	t_start = perf_counter()

	args = check_args(sys.argv)

	if args == -1:
			sys.exit(-1)
	
	disease_Id	= args[0]
	methods			= args[1]
	num_cpus		= args[2]
	datasets		= args[3]

	# Comment if not using GPU
	# multiprocessing.set_start_method('spawn', force=True)

	# Check if the number of cpus selected by the user is greater than the
	# number of cores of the machine, if so, set num_cpus to the maximum number
	# possible
	host_cpu_count = multiprocessing.cpu_count()
	if num_cpus > host_cpu_count:
			print('\t[i] Passed', num_cpus, 'as num_cores, but is seems that you have only', host_cpu_count,\
					'to avoid errors, num_cores is set to', host_cpu_count)
			num_cpus = host_cpu_count

	# if disease_Id != 'all':
	# 		disease_Ids = [disease_Id]
	
	if methods[0] == 'all':
		methods = available_methods

	if datasets != 'all':
		datasets = [datasets]
	else:
		datasets = available_datasets

	# Create a list to store the parameters that will
	# be mapped to the different processes
	params = []

	for dataset in datasets:
		diseases = disease_Id

		if disease_Id[0] == 'all':
			diseases = disgenet_disease_Ids if dataset == 'disgenet' else omim_disease_Ids
		# print('\n[i] Computing the ranking for', diseases, '(', len(diseases), ')', 'disease(s) on dataset', dataset)

		for disease in diseases:
			for method in methods:
				# print('\t[+] Starting', disease, 'with method', METHOD, 'on dataset', dataset)
				filename = PATH_TO_RANKINGS + disease + '_Phat_and_P_' + dataset + '_'

				modality = 'multiclass'
				if '_only' in method:
						modality = 'binary'
				
				if modality == 'multiclass':
						filename += 'xgdag_' + method.lower() + '.txt'
				else:
						filename += method.lower().replace("_only", "") + '.txt'

				res = ''
				if os.path.exists(filename):
					print('[i] Skipping disease', disease, 'with method', method)
					continue
				# 		res = sanitized_input('[+] A raking for disease ' + disease + \
				# 				' has already been computed with ' + METHOD + \
				# 				'. Do you want to overwrite the old ranking? (y|n) ', ['y', 'n'])
				# if res == 'n':
				else:
						# Compute the ranking
					args = [disease, method, num_cpus, filename, dataset, modality]
					ranking(args)
					# params.append(args)
	
	# with multiprocessing.Pool(num_cpus) as pool:
	# 	# Map the function to the parameters in parallel
	# 	pool.map(ranking, params)

	t_end = perf_counter()
	print('[i] Elapsed time:', round(t_end - t_start, 3))