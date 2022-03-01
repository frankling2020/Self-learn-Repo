##########################################################################
##                                IMPORT                                ##
##########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_sparse

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

import numpy as np
from tqdm import trange, tqdm
import pandas as pd
import copy

import matplotlib.pyplot as plt
from zmq import device



##########################################################################
##                                LAYER                                 ##
##########################################################################

class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, inputs, adj):
        out = torch_sparse.matmul(adj, inputs, reduce='add')
        return torch.mm(out, self.weight) + self.bias

    def reg_loss(self):
        return torch.sum(self.weight**2)

    def __repr__(self):
        return self.__class__.__name__ + '_from_' + \
            str(self.in_features) + '_to_' + str(self.out_features)



##########################################################################
##                                MODEL                                 ##
##########################################################################

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_dims, 
            num_layers, out_features, dropout, reg=0):
        super(GCN, self).__init__()
        assert num_layers >= 2
        
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.out_features = out_features
        self.dropout = dropout
        self.reg = reg

        self.layers = nn.ModuleList([GCNLayer(in_features, hidden_dims)])
        self.layers.extend([
            GCNLayer(hidden_dims, hidden_dims) for x in range(num_layers-1)
        ])
        self.post_mp = torch.nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dims, out_features))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.post_mp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        deg = torch_geometric.utils.degree(edge_index[0], data.num_nodes, dtype=x.dtype).to(x.device)
        value = 1/torch.sqrt(deg[edge_index[0]] *  deg[edge_index[1]])
        adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=value,
                   sparse_sizes=(data.num_nodes, data.num_nodes))
        out = x
        for layer in self.layers:
            out = F.relu(layer(out, adj))
            out = F.dropout(out, self.dropout, training=self.training)
        return F.log_softmax(self.post_mp(out), dim=1)

    def loss(self, pred, label):
        loss = F.nll_loss(pred, label)
        if self.reg != 0:
            for layer in self.layers:
                loss += self.reg*layer.reg_loss()
            for layer in self.post_mp:
                if isinstance(layer, nn.Linear):
                    loss += self.reg*torch.sum(layer.weight.data**2)
        return loss


##########################################################################
##                           TRAIN AND TEST                             ##
##########################################################################


def train(dataset, args):
    device = args.device
    print("Node task. test set size:", np.sum(dataset[0]['test_mask'].numpy()))
    print()
    test_loader = loader = DataLoader(dataset, 
                                batch_size=args.batch_size, shuffle=False)
    # build model
    model = GCN(dataset.num_node_features, args.hidden_dim, 
                args.num_layers, dataset.num_classes, 
                dropout=args.dropout, reg=args.reg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, 
                weight_decay=args.weight_decay)

    # train
    train_losses = []
    valid_accs = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch.to(device))
            label = batch.y.to(device)
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        train_losses.append(total_loss)

        if epoch % 10 == 0:
            valid_acc = test(loader, model, device, is_validation=True)
            valid_accs.append(valid_acc)
            test_acc = test(test_loader, model, device)
            test_accs.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
        else:
            valid_accs.append(valid_accs[-1])
            test_accs.append(test_accs[-1])
    
    return valid_accs, test_accs, train_losses, best_model, best_acc, test_loader

def test(loader, test_model, device, is_validation=False, 
        save_model_preds=False, model_type=None):
    
    test_model.eval()

    correct = 0
    # Note that Cora is only one graph!
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = test_model(data.to(device)).max(dim=1)[1]
            label = data.y.to(device)

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = label[mask]

        if save_model_preds:
            print ("Saving Model Predictions for Model Type", model_type)

            data = {}
            data['pred'] = pred.view(-1).cpu().detach().numpy()
            data['label'] = label.view(-1).cpu().detach().numpy()

            df = pd.DataFrame(data=data)
            # Save locally as csv
            df.to_csv('CORA-Node-' + model_type + '.csv', sep=',', index=False)
            
        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total
  


##########################################################################
##                                UTILS                                 ##
##########################################################################

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


##########################################################################
##                                UTILS                                 ##
##########################################################################

if __name__=="__main__":
    setup_seed(22)
    for args in [
        {
            'device': 'cuda:0', 
            'model_type': 'GCN', 
            'dataset': 'cora', 
            'num_layers': 2, 
            'batch_size': 32, 
            'hidden_dim': 32, 
            'dropout': 0.5,
            'epochs': 500,
            'weight_decay': 5e-3, 
            'lr': 1e-2,
            'reg': 5e-4
        },
    ]:
        args = objectview(args)
    
    model_type = args.model_type
    plt.style.use('seaborn-paper')

    if args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/cora', name='Cora')
    else:
        raise NotImplementedError("Unknown dataset") 
    valid_accs, test_accs, losses, best_model, best_acc, test_loader = train(dataset, args) 

    print("Maximum valid set accuracy: {0}".format(max(valid_accs)))
    print("Maximum test set accuracy: {0}".format(max(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))

    # Run test for our best model to save the predictions!
    test(test_loader, best_model, args.device, 
        is_validation=False, save_model_preds=True, model_type=model_type)
    print()
    
    
    plt.title(dataset.name)
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(valid_accs, label="valid accuracy" + " - " + args.model_type)
    plt.plot(test_accs, label="test accuracy" + " - " + args.model_type)
    plt.legend()
    plt.savefig(f"result_{model_type}.jpg", dpi=300)
    plt.close()

    torch.save(best_model, f"best_model_{model_type}.pth")