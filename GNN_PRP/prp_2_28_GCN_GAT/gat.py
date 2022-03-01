########################################################
##                       IMPORT                       ##
########################################################

import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.utils import softmax, degree
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange
import pandas as pd
import copy

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt



########################################################
##                      GAT MODEL                     ##
########################################################

class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., eps=1e-2, **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = eps

        self.lin_l = nn.Linear(self.in_channels, self.heads*self.out_channels)
        self.lin_r = self.lin_l

        self.attr = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.lin_l.weight)
        nn.init.xavier_normal_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.attr)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        
        out = self.propagate(edge_index=edge_index, size=size, x=(x_l, x_r))
        out = out.view(-1, H*C)

        return out

    def message(self, x_j, x_i, index):
        x = x_j + (1+self.eps) * x_i
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        alpha_ij = (x * self.attr).sum(dim=-1)
        alpha_ij = F.dropout(softmax(alpha_ij, index), p=self.dropout, inplace=True, training=self.training)
        out = x_j * alpha_ij.unsqueeze_(-1)
        return out

    def aggregate(self, inputs, index):
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, node_dim, reduce='sum')
        return out
  



########################################################
##                       GNN STACK                    ##
########################################################

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        self.heads = args.heads
        
        conv_model = GAT
        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim, heads=self.heads))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim, heads=self.heads))

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.post_mp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
          
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)



########################################################
##                       OPTIMIZER                    ##
########################################################

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



########################################################
##                    TRAIN AND TEST                  ##
########################################################

def train(dataset, args):
    device = args.device
    print("Node task. test set size:", np.sum(dataset[0]['test_mask'].numpy()))
    print()
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
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
        losses.append(total_loss)

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
    
    
    return valid_accs, test_accs, losses, best_model, best_acc, test_loader

def test(loader, test_model, device, is_validation=False, save_model_preds=False, model_type=None):
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
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d



########################################################
##                         SEED                       ##
########################################################

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



########################################################
##                         MAIN                       ##
########################################################

if __name__=="__main__":
    setup_seed(22)
    for args in [
        {
            'device': 'cuda:0', 
            'model_type': 'GAT', 
            'dataset': 'cora', 
            'num_layers': 2, 
            'batch_size': 64, 
            'hidden_dim': 32, 
            'dropout': 0.5, 
            'epochs': 500, 
            'opt': 'adam', 
            'opt_scheduler': 'none', 
            'opt_restart': 0, 
            'weight_decay': 5e-3, 
            'lr': 1e-3,
            'heads': 2,
        },
    ]:
        args = objectview(args)
    
    plt.style.use('seaborn-paper')

    model = 'GAT'
    args.model_type = model
    print("Multi-heads: ", args.heads)
    if args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/cora', name='Cora')
    else:
        raise NotImplementedError("Unknown dataset") 
    valid_accs, test_accs, losses, best_model, best_acc, test_loader = train(dataset, args) 

    print("Maximum valid set accuracy: {0}".format(max(valid_accs)))
    print("Maximum test set accuracy: {0}".format(max(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))

    # Run test for our best model to save the predictions!
    test(test_loader, best_model, args.device, is_validation=False, save_model_preds=True, model_type=model)
    print()

    plt.title(dataset.name)
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_accs, label="test accuracy" + " - " + args.model_type)
    plt.legend()
    plt.savefig(f'result_{model}.jpg', dpi=300)
    plt.close()
    
    torch.save(best_model, f"best_model_{model}.pth")