import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import numpy as np

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator, RFEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt
from torch_geometric.data import Data


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))
        # self.reset_parameters()

    @staticmethod
    def linear_reset(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)
        elif isinstance(layer, nn.BatchNorm1d):
            nn.init.ones_(layer.weight.data)

    def reset_parameters(self):
        for layer in self.project:
            self.linear_reset(layer)
        for conv in self.layers:
            for layer in conv.nn:
                self.linear_reset(layer)

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

    def reg_loss(self):
        loss = 0
        for params in self.parameters():
            if params.requires_grad and len(params.shape) > 1:
                loss += torch.sum(params**2)
        return loss

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, _ = aug1(x, edge_index)
        x2, edge_index2, _ = aug2(x, edge_index)

        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        gs3 = []
        gs4 = []
        for num in range(3):
            edge_index3, _ = A.functional.random_walk_subgraph(edge_index, None, batch_size=1000, length=7+num)
            edge_index4, _ = A.functional.random_walk_subgraph(edge_index, None, batch_size=999, length=12+num)
            _, g3 = self.encoder(x, edge_index3, batch)
            _, g4 = self.encoder(x, edge_index4, batch)
            gs3.append(g3)
            gs4.append(g4)
        return z, g, z1, z2, g1, g2, x1, x2, gs3, gs4
    def reg_loss(self):
        return self.encoder.reg_loss()


def train(encoder_model, contrast_model, dataloader, optimizer, alpha=0, reg=0):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2, x1, x2, gs3, gs4 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch) - alpha * contrast_model(g1=x1, g2=x2, batch=data.batch)
        loss += 0.5*torch.tensor([contrast_model(g1=g3, g2=g4, batch=data.batch) for g3, g4 in zip(gs3, gs4)], device=data.x.device).mean()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator()(x, y, split)
    return result


def main():
    # setup_seed(22)
    torch.cuda.empty_cache()
    datasets_name = 'NCI1'
    batch_size = 64
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('.'), 'datasets')
    dataset = TUDataset(path, name=datasets_name)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.Compose([
                A.RandomChoice([
                    A.NodeDropping(pn=0.15),
                    A.FeatureMasking(pf=0.15),
                    A.EdgeRemoving(pe=0.15),
                ], 1),
                # A.NodeShuffling(),
            ])
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=1e-2, weight_decay=1e-3)

    epochs = 100
    alpha = 0.1
    reg = 1e-3
    losses = []
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, 1+epochs):
            loss = train(encoder_model, contrast_model, dataloader, optimizer, alpha, reg)
            losses.append(loss)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.3f}, F1Ma={test_result["macro_f1"]:.3f}')

    plt.title(f'{batch_size} batch-sized {datasets_name} with \n Best test accuracy= {test_result["accuracy"]:.3f}, test F1Mi={test_result["micro_f1"]:.3f}, F1Ma={test_result["macro_f1"]:.3f}')
    plt.plot(losses, label="training loss")
    plt.legend()
    plt.savefig(f"result_{datasets_name}.jpg", dpi=300)
    plt.close()
    print("Plot!")

    torch.save(encoder_model, f"encoder_mdl_{datasets_name}.pth")
    print("Save!")


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()