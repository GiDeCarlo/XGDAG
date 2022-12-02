### Module file
from GraphSageModel import GNN7L_Sage
from Paths import PATH_TO_IMAGES, PATH_TO_REPORTS, PATH_TO_MODELS

import pandas as pd
import seaborn as sn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, data, epochs, lr, weight_decay, classes, model_name):    
    data = data.to(device)
    model = model.to(device)

    title = model_name + '_' + str(epochs) + '_' + str(weight_decay).replace('.', '_')

    model_path  = PATH_TO_MODELS + title

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask  = data['train_mask']
    val_mask    = data['val_mask']

    labels = data.y
    output = ''

    # list to plot the train accuracy
    train_acc_curve = []
    train_lss_curve = []

    best_train_acc  = 0
    best_val_acc    = 0
    best_train_lss  = 999
    best_loss_epoch = 0

    for e in tqdm(range(epochs+1)):
        model.train()
        optimizer.zero_grad()
        logits      = model(data.x, data.edge_index)
        output      = logits.argmax(1)
        # train_loss  = F.cross_entropy(logits[train_mask], labels[train_mask])
        train_loss  = F.nll_loss(logits[train_mask], labels[train_mask])
        train_acc   = (output[train_mask] == labels[train_mask]).float().mean()
        train_loss.backward()
        optimizer.step()

        # Append train acc. to plot curve later
        train_acc_curve.append(train_acc.item())
        train_lss_curve.append(train_loss.item())

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        # Evaluation and test
        model.eval()
        logits      = model(data.x, data.edge_index)
        output      = logits.argmax(1)
        val_loss    = F.nll_loss(logits[val_mask], labels[val_mask])
        val_acc     = (output[val_mask] == labels[val_mask]).float().mean()

        # Update best test/val acc.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Save model with best train loss
        if train_loss < best_train_lss:
            best_train_lss = train_loss
            best_loss_epoch = e
            torch.save(model.state_dict(), model_path)

        if e % 20 == 0 or e == epochs:
            print('[Epoch: {:04d}]'.format(e),
            'train loss: {:.4f},'.format(train_loss.item()),
            'train acc: {:.4f},'.format(train_acc.item()),
            'val loss: {:.4f},'.format(val_loss.item()),
            'val acc: {:.4f} '.format(val_acc.item()),
            '(best train acc: {:.4f},'.format(best_train_acc.item()),
            'best val acc: {:.4f},'.format(best_val_acc.item()),
            'best train loss: {:.4f} '.format(best_train_lss),
            '@ epoch', best_loss_epoch ,')')
    
    # Plot training accuracy curve
    plt.figure(figsize = (12,7))
    plt.plot(train_acc_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.show()

    plt.figure(figsize = (12,7))
    plt.plot(train_lss_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.show()

    predict_from_saved_model(title, data, classes)

    return output

def predict_from_saved_model(model_name, data, classes, files_name='', plot_results=True, save_to_file=True):

    if not plot_results and save_to_file:
        print('[i] plot_results set to', plot_results, 'but save_to_file set to', save_to_file)
        print('with such configuration, only the report will be saved but not the confusion matrices')

    data = data.to(device)
    
    model_path  = PATH_TO_MODELS + model_name
   
    if files_name != '':
        image_path  = PATH_TO_IMAGES + files_name
        report_path = PATH_TO_REPORTS + files_name + '.csv'
    else:
        image_path  = PATH_TO_IMAGES + model_name
        report_path = PATH_TO_REPORTS + model_name + '.csv'

    test_mask   = data['test_mask']

    labels = data.y

    # Load best model
    loaded_model = GNN7L_Sage(data)
    
    loaded_model = loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    loaded_model.eval()
    logits = loaded_model(data.x, data.edge_index)
    output = logits.argmax(1)

    if plot_results:
        print(classification_report(labels[test_mask].to('cpu'), output[test_mask].to('cpu')))

    if save_to_file:
        class_report = classification_report(labels[test_mask].to('cpu'), output[test_mask].to('cpu'), output_dict=True)
        classification_report_dataframe = pd.DataFrame(class_report)
        classification_report_dataframe.to_csv(report_path)

    #Confusion Matrix
    if plot_results:
        norms = [None, "true"]
        for norm in norms:
            cm = confusion_matrix(labels[test_mask].to('cpu'), output[test_mask].to('cpu'), normalize=norm)

            plt.figure(figsize=(7,7))
            
            if norm == "true":
                sn.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'BuPu', xticklabels = classes, yticklabels = classes)
            else:
                sn.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'BuPu', xticklabels = classes, yticklabels = classes)
            plt.title(model_name if files_name == '' else files_name)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')

            if save_to_file:
                if norm == None:
                    plt.savefig(image_path + '_notNorm.png', dpi=300)
                else:
                    plt.savefig(image_path + '_Norm.png', dpi=300)
    
    return output, logits, loaded_model