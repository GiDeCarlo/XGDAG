
import yaml
from time import perf_counter

from src.GNNTrain import train
from src.CreateDataset import get_dataset_from_graph
from src.GraphSageModel import GNN7L_Sage

def trainGNN(disease_Id, mode='multiclass', seed = 42, from_diamond=False, graph_path=None, dataset_path = None, epochs=1000, lr=0.001, weight_decay=0.0005, hidden_channels=16, quartiles=False):
    
    if mode == 'multiclass':
        classes     = ['P', 'LP', 'WN', 'LN', 'RN']
    elif mode == 'binary':
        classes = ['P', 'U']
    else:
        raise ValueError('Invalid mode. Only multiclass and binary are supported.')

    model_name  = 'GraphSAGE_' + disease_Id
    if from_diamond:
        model_name += '_diamond_data'

    if not from_diamond:
        graph_path += disease_Id + '.gml'
    else:
        graph_path += '_diamond_' + disease_Id + '.gml'

    dataset = None
    G       = None

    if mode == 'binary':
        model_name += '_binary'

    #dataset, G = CreateDatasetv2_binary_diamond.get_dataset_from_graph(graph_path, disease_Id, quartile=False)
    dataset, G = get_dataset_from_graph(graph_path, dataset_path, disease_Id, mode = mode, quartiles=quartiles, from_diamond=from_diamond, seed=seed)

    # model = GNN7L_Sage(dataset, hidden_channels=HIDDEN_CHANNELS)

    # preds = train(model, dataset, epochs, lr, weight_decay, classes, model_name)

if __name__ == '__main__':

    args = None
    
    with open("parameters.yml") as paramFile:  
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

    DISEASE_IDs  = args["trainer_script"]["DISEASE_IDs"]
    SEED = args["trainer_script"]["SEED"]
    MODE = args["trainer_script"]["MODE"]
    FROM_DIAMOND = args["trainer_script"]["FROM_DIAMOND"]
    GRAPH_PATH =  args["trainer_script"]["GRAPH_PATH"]
    LEARNING_RATE = args["trainer_script"]["LEARNING_RATE"]
    EPOCHS = args["trainer_script"]["EPOCHS"]
    WEIGHT_DECAY = args["trainer_script"]["WEIGHT_DECAY"]
    HIDDEN_CHANNELS = args["trainer_script"]["HIDDEN_CHANNELS"]
    DATASET_PATH = args["trainer_script"]["DATASET_PATH"]
    QUARTILES = args["trainer_script"]["QUARTILES"]

    for disease_Id in DISEASE_IDs:

        for mode in MODE:
            print('[+] Training ', disease_Id, ' in mode ', mode)
            start = perf_counter()
            trainGNN(disease_Id, mode=mode, seed=SEED, from_diamond=FROM_DIAMOND, graph_path=GRAPH_PATH, dataset_path = DATASET_PATH, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, hidden_channels=HIDDEN_CHANNELS, quartiles=QUARTILES)
            print('[+] Done ', disease_Id, ' in mode ', mode)
            end = perf_counter()
            print('[+] Time taken: ', round(end - start, 3), ' seconds.')
   