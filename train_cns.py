import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomNodeSplit 
import gc, sys, time
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import SparseTensor
from layers import CEConv
from utils import get_normed_lapacian, sp_laplacian_expo, get_sin, get_cos, get_filter
from torch_geometric.datasets import MixHopSyntheticDataset, CoraFull, Amazon, Coauthor,LastFMAsia,DBLP,CitationFull
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models.correct_and_smooth import CorrectAndSmooth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#dataset = Planetoid(root='./datasets', name='citeseer')
dataset = Planetoid(root='./datasets', name='cora')
#dataset = Amazon(root='./datasets', name='Photo')
#dataset = Coauthor(root='./datasets', name='CS', transform=T.RandomNodeSplit())
#dataset = DBLP(root='./datasets')
#dataset = CitationFull(root='./datasets', name='DBLP', transform=T.RandomNodeSplit())

#k_filter = 3 + 1
k_filter = 4
ita = 1e-3


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = CEConv(in_channels=dataset.num_node_features, out_channels=64, K=k_filter,
                              cos_list=cos_list, sin_list=sin_list)
        self.conv2 = CEConv(in_channels=64, out_channels=dataset.num_classes, K=k_filter,
                              cos_list=cos_list, sin_list=sin_list)

    def forward(self, edge_index, edge_weight, x):
        x = self.conv1(edge_index, edge_weight, x)
        x = F.relu(x)
        x = F.dropout(x, p=0.75, training=self.training)  # training -- apply dropout if is True. Default: True
        x = self.conv2(edge_index, edge_weight, x)

        return F.log_softmax(x, dim=1)  # dim=1

@torch.no_grad()
def test(out=None):
    model.eval()

    out = model(edge_index, edge_weight, x) if out is None else out
    _, pred = out.max(dim=1)  # model(data)---->torch.Size([2708, 7])
    
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    
    val_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    val_acc = val_correct / int(data.val_mask.sum())
    
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())


    return train_acc, val_acc, test_acc, out

if __name__ == '__main__':

    data = dataset[0].to(device)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    edge_index, edge_weight = get_normed_lapacian(edge_index, normalization='sym')  # eigvalues \in [-1, 1]

    cos_list, sin_list = get_filter(edge_index, edge_weight, k_taylor=19, k_filter=k_filter)  # 20 orders

    model = Net().to(device)
    print(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.Adam([
        {'params': model.conv1.weight, 'weight_decay': 5e-4},
        {'params': model.conv1.bias, 'weight_decay': 5e-4},
        {'params': model.conv1.a, 'weight_decay': 0},
        {'params': model.conv1.b, 'weight_decay': 0},
        {'params': model.conv2.weight, 'weight_decay': 5e-4},
        {'params': model.conv2.bias, 'weight_decay': 5e-4},
        {'params': model.conv2.a, 'weight_decay': 0},
        {'params': model.conv2.b, 'weight_decay': 0}
    ], lr=0.01)
    
    adj_t = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
    adj_t = adj_t.to_symmetric() 
    adj_t = adj_t.to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    y_train = data.y[data.train_mask]
    
    res = list()
    best_val_acc = 0

    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()
        y_hat = model(edge_index, edge_weight, x)
        l1loss_ab = torch.norm(model.conv1.a, 1) + torch.norm(model.conv1.b, 1) + torch.norm(model.conv2.a, 1) \
                    + torch.norm(model.conv2.b, 1)
        # loss = F.nll_loss(y_hat[data.train_mask], data.y[data.train_mask])
        loss = F.nll_loss(y_hat[data.train_mask], data.y[data.train_mask]) + ita * l1loss_ab
        loss.backward()
        optimizer.step()

        model.eval()
        # Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor
        # in the given dimension dim. And indices is the index location of each maximum value found (argmax).
        #out = model(edge_index, edge_weight, x)
        #_, pred = out.max(dim=1)  # model(data)---->torch.Size([2708, 7])
        

        #correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        #acc = correct / int(data.test_mask.sum())
        
        train_acc, val_acc, acc, out = test()
        if epoch % 10 == 0:
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Test acc: {:.4f}'
            print(log.format(epoch, loss.item(), acc))
            
        if acc > best_val_acc:
            best_val_acc = acc
            y_soft = out.softmax(dim=-1)

        if epoch % 100 == 0:
            print('Correct and smooth...')
            post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=0.8,
                                    num_smoothing_layers=50, smoothing_alpha=0.8,
                                    autoscale=False, scale=1.)

            y_soft = post.correct(y_soft, y_train, data.train_mask, DAD)
            y_soft = post.smooth(y_soft, y_train, data.train_mask, DAD)

            train_acc, val_acc, acc, _ = test(y_soft)
            print(f'Train: {train_acc:.4f}, Val:{val_acc:.4f}  Test: {acc:.4f}')
            
        res.append(acc)
               
    print('Best accuracy is: {:.4f}'.format(max(res)))

