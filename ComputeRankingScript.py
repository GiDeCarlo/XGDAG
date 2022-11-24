from GNNTrain import predict_from_saved_model
from CreateDatasetv2 import get_dataset_from_graph
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
from GDARanking import predict_candidate_genes

import sys

args = sys.argv

if len(sys.arg) < 2:
    print('Insert DiseaseId')

disease_Id = args[1]
METHOD = [2]


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