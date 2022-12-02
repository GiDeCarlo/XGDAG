from GNNTrain import predict_from_saved_model
from CreateDatasetv2 import get_dataset_from_graph
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
from GDARanking import predict_candidate_genes
import CreateDatasetv2_binary 

import os
import sys
from time import perf_counter

disease_Ids = ['C0006142',  'C0009402', 'C0023893', \
               'C0036341',  'C0376358', 'C3714756', \
               'C0860207',  'C0011581', 'C0005586', \
               'C0001973']

methods = ['gnnexplainer',  'gnnexplainer_only',\
           'graphsvx',      'graphsvx_only',    \
           'subgraphx',     'subgraphx_only']

def check_args(args):
    if len(args) < 3:
        if len(args) == 2:
            if args[1] == '-h' or args[1] == '--help':
                print('\n\n[Usage]: python ComputeRankingScript.py disease_id explainability_method num_cores\n')
                print('Available diseases:\n \tC0006142\n \tC0009402\n \tC0023893\n \tC0036341\n \tC0376358\tType all to compute the ranking for all the available diseases\n')
                print('Available methods:\n \tgnnexplainer\n \tgnnexplainer_only\n \tgraphsvx\n \tgraphsvx_only\n \tsubgraphx\n')
                print('num_cores: define the number of cores to parallelize the explainability method\n\n')
        else:
            print('\n\n[ERR] Wrong input parameters: use -h or --help to print the usage\n\n')
        return -1

    disease_Id = args[1]
    METHOD = args[2]
    num_cpus = int(args[3])

    if disease_Id not in disease_Ids and disease_Id.lower() != 'all':
        print('\n[ERR] Disease ID', disease_Id, 'not present in the database\n')
        return -1

    if METHOD not in methods and METHOD.lower() != 'all':
        print('\n[ERR] Method', METHOD, 'not available\n')
        return -1
    
    if num_cpus < 1:
        print('\n[ERR]', num_cpus,'is an nvalid number of cores\n')
        return -1
    
    return disease_Id, METHOD, num_cpus

def ranking(disease_Id, METHOD, num_cpus, filename, modality='multiclass'):

    model_name  = 'GraphSAGE_' + disease_Id + '_new_rankings'
    graph_path  = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'
    classes     = ['P', 'LP', 'WN', 'LN', 'RN']

    if modality == 'binary':
        model_name += '_binary'
        classes = ['P', 'U']
        dataset, G = CreateDatasetv2_binary.get_dataset_from_graph(graph_path, disease_Id, quartile=False)
    else:
        dataset, G = get_dataset_from_graph(graph_path, disease_Id, quartile=False)

    model_name += '_40000_0_0005'

    preds, probs, model = predict_from_saved_model(model_name, dataset, classes, save_to_file=False)

    ranking = predict_candidate_genes(model,
                                    dataset,
                                    preds,
                                    explainability_method=METHOD,
                                    disease_Id=None,
                                    explanation_nodes_ratio=1,
                                    num_hops=1,
                                    G=G,
                                    num_pos="all",
                                    num_workers=num_cpus)

    print('[+] Saving ranking to file', filename, end='...')

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
    
    disease_Id = args[0]
    METHOD = args[1]
    num_cpus = args[2]

    modality = 'multiclass'
    if '_only' in METHOD:
        modality = 'binary'

    if disease_Id != 'all':
        disease_Ids = [disease_Id]
    
    if METHOD != 'all':
        methods = [METHOD]
    
    print('[i] Computing the ranking for', disease_Ids, '(', len(disease_Ids), ')', 'disease(s).')
    
    for disease_Id in disease_Ids:
        print('[i] Starting', disease_Id)

        filename = PATH_TO_RANKINGS + disease_Id + '_all_positives_new_ranking_'

        for METHOD in methods:
            if modality == 'multiclass':
                filename += 'xgdag_' + METHOD.lower() + '.txt'
            else:
                filename += METHOD.lower().replace("_only", "") + '.txt'

            res = ''
            if os.path.exists(filename):
                res = sanitized_input('[+] A raking for disease ' + disease_Id + \
                    ' has already been computed with ' + METHOD + \
                    '. Do you want to overwrite the old ranking? (y|n) ', ['y', 'n'])
            
            if res == 'n':
                print('[i] Skipping disease', disease_Id)
                continue

            ranking(disease_Id, METHOD, num_cpus, filename, modality)

    t_end = perf_counter()
    print('[i] Elapsed time:', round(t_end - t_start, 3))