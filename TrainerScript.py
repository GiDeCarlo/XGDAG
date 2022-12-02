from GNNTrain import train, predict_from_saved_model
from CreateDatasetv2 import get_dataset_from_graph
from Paths import PATH_TO_GRAPHS, PATH_TO_RANKINGS
from GDARanking import get_ranking, predict_candidate_genes, validate_with_extended_dataset, get_ranking_no_LP_intersection, validate_with_extended_dataset_no_LP
from GraphSageModel import GNN7L_Sage
import CreateDatasetv2_binary

def trainGNN(disease_Id, mode='binary'):
    
    classes     = ['P', 'LP', 'WN', 'LN', 'RN']
    model_name  = 'GraphSAGE_' + disease_Id + '_new_rankings'
    graph_path  = PATH_TO_GRAPHS + 'grafo_nedbit_' + disease_Id + '.gml'

    dataset = None
    G       = None

    if mode == 'binary':
        classes = ['P', 'U']
        model_name += '_binary'
        dataset, G = CreateDatasetv2_binary.get_dataset_from_graph(graph_path, disease_Id, quartile=False)
    else:
        dataset, G = get_dataset_from_graph(graph_path, disease_Id, quartile=False)

    lr              = 0.001
    epochs          = 40000
    weight_decay    = 0.0005

    model = GNN7L_Sage(dataset)

    preds = train(model, dataset, epochs, lr, weight_decay, classes, model_name)

if __name__ == '__main__':

    disease_Ids = ['C3714756','C0860207','C0011581','C0005586','C0001973']

    for disease_Id in disease_Ids:
        print('[+] Training', disease_Id)
        trainGNN(mode='binary')
        trainGNN(mode='multiclass')
