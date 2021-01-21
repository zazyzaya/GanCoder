import torch 
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GCNConv

class GAE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32, use_adj_recon_loss=False):
        super(GAE, self).__init__()

        self.c1 = GCNConv(feat_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.sig = nn.Sigmoid()

        self.embed_dim = embed_dim
        self.use_adj_recon_loss = use_adj_recon_loss


    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)
        x = self.drop(x)

        return x

    def recon(self, x, ei, ew=None):
        z = self.forward(x, ei, ew=ew)
        return self.sig(torch.mm(z,z.T))

    def decode(self, z, ei):
        dot = (z[ei[0]] * z[ei[1]]).sum(dim=1)
        return self.sig(dot)

    def loss_fn(self, *args):
        if self.use_adj_recon_loss:
            return self.adj_recon_loss(args[0], args[1])
        else:            
            return self.per_edge_loss(args[0], args[1], args[2])

    '''
    This is the default loss function used by the paper. It may be
    good for feature prediction, I'm not sure, but using per-edge
    loss is waaaay better for link prediction
    '''
    def adj_recon_loss(self, z, pos_samples):        
        adj_pr = self.sig(torch.mm(z,z.T))
        adj = torch.zeros(adj_pr.size())
        adj[pos_samples[0], pos_samples[1]] = 1

        norm = (adj.size(0) ** 2) / ((adj.size(0) ** 2 - adj.sum()) * 2)
        p_weight = (adj.size(0) ** 2 - adj.sum()) / adj.sum()

        loss = F.binary_cross_entropy_with_logits(
            adj_pr, adj,
            pos_weight=p_weight
        )

        return norm * loss 

    '''
    Rather than trying to rebuild the entire adj matrix, just test
    if the dot product of specific edges is 1 and the dot product of
    specific negative edges is close to 0. 

    Far more scalable, and produces much higher AUC and APs
    '''
    def per_edge_loss(self, z, pos_samples, neg_samples):
        EPS = 1e-6
        pos_loss = -torch.log(self.decode(z, pos_samples)+EPS).mean()

        neg_loss = -torch.log(1-self.decode(z, neg_samples)+EPS).mean()
        return pos_loss + neg_loss
