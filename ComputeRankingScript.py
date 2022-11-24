from GNNTrain import predict_from_saved_model
from CreateDatasetv2 import get_dataset_from_graph
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
from GDARanking import predict_candidate_genes

import sys

disease_Ids = ['C0006142', 'C0009402', 'C0023893', 'C0036341', 'C0376358']
methods = ['gnnexplainer', 'gnnexplainer_only', 'graphsvx', 'graphsvx_only', 'subgraphx']

args = sys.argv
if len(args) < 3:
    if args[1] == '-h' or args[1] == '--help':
        print('Available diseases:\n \tC0006142\n \tC0009402\n \tC0023893\n \tC0036341\n \tC0376358')
        print('Available methods:\n \gnnexplainer\n \gnnexplainer_only\n \graphsvx\n \graphsvx_only\n \subgraphx')
    else:
        print('[ERR] Wrong usage')
    print('Insert DiseaseId')

disease_Id = args[1]
METHOD = args[2]

if disease_Id not in disease_Ids:
    print('[ERR] Wrong disease ID')

if METHOD not in methods:
    print('[ERR] Method')


classes     = ['P', 'LP', 'WN', 'LN', 'RN']
model_name  = 'GraphSAGE_' + disease_Id + '_new_rankings'
graph_path  = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'

dataset, G = get_dataset_from_graph(graph_path, disease_Id, quartile=False)

preds, probs, model = predict_from_saved_model(model_name + '_40000_0_0005', dataset, classes, save_to_file=False)

ranking = predict_candidate_genes(model,
                                  dataset,
                                  preds,
                                  explainability_method=METHOD,
                                  disease_Id=None,
                                  explanation_nodes_ratio=1,
                                  num_hops=1,
                                  G=G,
                                  num_pos="all")

filename = PATH_TO_RANKINGS + disease_Id + '_all_positives_new_ranking_xgdag_' + METHOD.lower() + '.txt'
with open(filename, 'w') as f:
     for line in ranking:
        f.write(line + '\n')