import torch 
import numpy as np

from torch import nn
from torch_cluster import random_walk
from sklearn.metrics import roc_auc_score, average_precision_score

def get_score(tp_edges, neg, recon_adj, resize_neg=True):
    # Select an equal number of random false edges
    n = tp_edges.size(1)
    
    if resize_neg:
        neg = neg[:, torch.randperm(neg.size(1))[:n]]

    pscore = recon_adj[tp_edges[0], tp_edges[1]]
    nscore = recon_adj[neg[0], neg[1]]
    score = torch.cat([pscore, nscore]).numpy()

    labels = np.zeros(n*2, dtype=np.long)
    labels[:n] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return auc, ap

from sklearn.decomposition import PCA
def pca(X, dim=256):
    n_components = min(dim, X.size()[1])
    decomp = PCA(n_components=n_components, random_state=1337)
    return torch.tensor(decomp.fit_transform(X.numpy()))


'''
I'm not sure what the paper is on about with the whole
Pv = dv^(3/4) distribution thing. I'm just using the uniform
distro

Assumes self loops are added so torch.multinomial doesn't break
'''
def sample_triplets(adj, x, ei, batch=None):
    if type(batch) == type(None):
        edges = adj.float()
        batch = x
    
    else:
        edges = adj[batch].float()
        batch = x[batch]
    
    pos = torch.multinomial(
        edges, 
        num_samples=1, 
        replacement=True
    ).squeeze(-1)

    # If one node is connected to every other one, I 
    # think there are bigger problems
    neg = torch.multinomial(
        (edges == 0).float(),
        num_samples=1,
        replacement=True
    ).squeeze(-1)

    return batch, x[pos], x[neg]

'''
Transforms rw of Batch x Walk Len to tuples of original node 
and walked to node
'''
def __reshape_batch(x):
    if x.size(1) == 2:
        return x 
    
    origin, walks = x[:,0:1], x[:, 1:]
    origin = origin.repeat_interleave(walks.size(1), 0)
    walks = walks.reshape((walks.size(0)*walks.size(1), 1))

    return origin.squeeze(-1), walks.squeeze(-1)

WALK_LEN = 4
def sample_triplets_from_rw(adj, x, ei, batch=None):
    global WALK_LEN

    if type(batch) == type(None):
        edges = adj.float()
        batch = torch.tensor(list(range(x.size(0))))
    else:
        edges = adj[batch].float()

    pos = random_walk(ei[0], ei[1], batch, WALK_LEN)
    neg = torch.multinomial(
        (edges == 0).float(),
        num_samples=WALK_LEN,
        replacement=True
    ).squeeze(-1)
    # Retains the order, so [[a,b],[c,d]] becomes [[a,b,c,d]].T
    neg = neg.reshape((neg.size(1)*neg.size(0), 1)).squeeze(-1)

    origin, walks = __reshape_batch(pos)
    return x[origin], x[walks], x[neg]
    
'''
Iteratively build out min distance matrix
Exp is used to smooth output scores 
(1/x is too steep, the x^(3/4) relationship is well known in edge distro so we 
choose that)
'''
def rw_distance(x, ei, batch=None, walk_len=5, dist_mat=None, exp=0.75):
    if type(batch) == type(None):
        batch = torch.tensor(list(range(x.size(0))))
    
    if type(dist_mat) == type(None):
        dist_mat = torch.zeros((x.size(0), x.size(0)))

    rw = random_walk(ei[0], ei[1], batch, walk_len)
    for i in range(1, rw.size(1)):
        score = 1 / (i ** exp)

        for j in range(rw.size(1)-i):
            # Find more efficient way to do this
            vals = dist_mat[rw[:, j], rw[:, j+i]]
            vals[vals < score] = score
            dist_mat[rw[:, j], rw[:, j+i]] = vals

    return dist_mat


'''
    Impliments Minibatch Discrimination to avoid same-looking output
    Shamelessly stolen from https://gist.github.com/t-ae/
'''
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims=64, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # Outputs Batch x Out*Kernel 
        matrices = x.mm(self.T.view(self.in_features, -1))

        # Transforms to Batch x Out x Kernel
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        # Now we quickly find distance from each X to every other
        # X by viewing it as a 1 x Batch x Out x Kernel mat and a
        #                      Batch x 1 x Out x Kernel mat 
        # That way the difference along the kernel dimension is 
        # equivilant to the dist from x to every other sample
        M = matrices.unsqueeze(0)  
        M_T = M.permute(1, 0, 2, 3) 

        # Simple distance formula
        norm = torch.abs(M - M_T).sum(3)  # Batch x Batch x Out
        expnorm = torch.exp(-norm)
        
        # Add all distances together, and remove self distance (minus 1)
        o_b = (expnorm.sum(0) - 1)   # Batch x Out 
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x