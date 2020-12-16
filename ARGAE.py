import torch

from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

'''
Trying to impliment Adversarially Regularized GAE from
Pan et al., 2018 -- IJCAI
'''
class ARGAEGenerator(nn.Module):
    def __init__(self, feat_dim, latent_dim=16, hidden_dim=32):
        super(ARGAEGenerator, self).__init__()

        self.c1 = GCNConv(feat_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, latent_dim)
        self.sig = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        return self.c2(x, ei, edge_weight=ew)

    def recon(self, x, ei, ew=None):
        z = self.forward(x, ei, ew=ew)
        return self.sig(torch.mm(z,z.T))

    def loss_fn(self, inputs, target):
        adj = self.sig(torch.mm(inputs, inputs.T))
        return self.loss(adj, target)

class ARGAEDiscriminator(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64):
        super(ARGAEDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.feat_dim = feat_dim
        self.loss = nn.BCELoss()

        self.tl = 1.0
        self.fl = 0.0

    '''
    Allows either tensor of Zs to be passed in, or an int
    denoting the batch size of noise data to process 
    '''
    def forward(self, data):
        if type(data) == torch.Tensor:
            return self.layers(data)

        # Paper samples "true" values from normal distro
        x = Variable(
            torch.empty(data, self.feat_dim).normal_()
        )

        '''
        dist = torch.distributions.exponential.Exponential(1.0)
        x = Variable(dist.sample((data, self.feat_dim)))
        '''

        return self.layers(x)

    def loss_fn(self, inputs, tl):
        val = self.tl if tl else self.fl 
        labels = Variable(
            torch.full((inputs.size(0), 1), val)
        )

        return self.loss(inputs, labels)