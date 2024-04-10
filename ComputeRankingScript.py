from src.GNNTrain import predict_from_saved_model
from src.CreateDataset import get_dataset_from_graph
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
from GDARanking import predict_candidate_genes
from src.CreateDataset import get_dataset_from_graph

import os
import sys
from time import perf_counter
import yaml


available_methods = ['gnnexplainer', 'graphsvx',  'subgraphx']

def check_args(METHODS, mode, num_cpus):


    for METHOD in METHODS:
        if METHOD not in available_methods:
            print('\n[ERR] Method', METHOD, 'not available.\n')
            sys.exit(-1)
    
    if mode not in ['multiclass', 'binary']:
        print('\n[ERR] Mode', mode, 'not available. Only binary and multiclass supported.\n')
        sys.exit(-1)

    if num_cpus < 1:
        print('\n[ERR]', num_cpus,'is an nvalid number of cores.\n')
        sys.exit(-1)
    
    return None

def ranking(disease_Id, METHOD, num_cpus, filename, modality='multiclass', from_diamond=False, model_name_prefix='GraphSAGE_', dataset_path=None, graph_path=None, save_ranking_path=None, quartiles=False):

    model_name  = model_name_prefix + disease_Id
    if from_diamond:
        model_name += '_diamond'

    graph_path  = GRAPH_PATH + disease_Id + '.gml'

    classes     = ['P', 'LP', 'WN', 'LN', 'RN']

    if modality == 'binary':
        model_name += '_binary'
        classes = ['P', 'U']


    dataset, G = get_dataset_from_graph(graph_path, dataset_path, disease_Id, modality, quartiles, from_diamond=from_diamond)

    #model_name += '_40000_0_0005' TO REMOVE AND RENAME THE MODEL

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

    args = None
    
    with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

    DISEASE_IDs = args["compute_ranking_script"]["DISEASE_IDs"]
    EXPLAINERS = args["compute_ranking_script"]["EXPLAINERS"]
    MODE = args["compute_ranking_script"]["MODE"]
    NUM_CPUs = args["compute_ranking_script"]["NUM_CPUs"]
    FROM_DIAMOND = args["compute_ranking_script"]["FROM_DIAMOND"]
    GRAPH_PATH = args["compute_ranking_script"]["GRAPH_PATH"]
    SAVE_RANKING_PATH = args["compute_ranking_script"]["SAVE_RANKING_PATH"]
    MODEL_NAME_PREFIX = args["compute_ranking_script"]["MODEL_NAME_PREFIX"]

    t_start = perf_counter()

    check_args(EXPLAINERS, MODE, NUM_CPUs)
    
    print('[i] Computing the ranking for', DISEASE_IDs, '(', len(DISEASE_IDs), ')', 'disease(s).')
    
    for disease_Id in DISEASE_IDs:

        for METHOD in EXPLAINERS:
            print('[i] Starting', disease_Id, 'with method', METHOD)

            for mode in MODE:
                print('[i] Starting', disease_Id, 'with method', METHOD, 'in mode', mode)

                filename = SAVE_RANKING_PATH + disease_Id # '_all_positives_diamond_'
                if FROM_DIAMOND:
                    filename += '_diamond_'
                else:
                    filename += '_'
                
                if MODE == 'multiclass':
                    filename += 'xgdag_' + METHOD.lower() + '.txt'
                else:
                    filename += 'xgdag_' + METHOD.lower() + '_binary.txt'

                res = ''
                if os.path.exists(filename):
                    res = sanitized_input('[+] A raking for disease ' + disease_Id + \
                        ' has already been computed with ' + METHOD + \
                        '. Do you want to overwrite the old ranking? (y|n) ', ['y', 'n'])
                
                if res == 'n':
                    print('[i] Skipping disease', disease_Id, 'with method', METHOD)
                    continue
                else:
                    # Compute the ranking
                    ranking(disease_Id, METHOD, NUM_CPUs, filename, mode, FROM_DIAMOND, MODEL_NAME_PREFIX)

    t_end = perf_counter()
    print('[i] Elapsed time:', round(t_end - t_start, 3))