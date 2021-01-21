import torch 
import numpy as np

from torch import nn
from torch_cluster import random_walk
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.metrics import roc_auc_score, average_precision_score

def decode(z, ei):
    dot = (z[ei[0]] * z[ei[1]]).sum(dim=1)
    return torch.sigmoid(dot)

def get_score(tp, neg, z):
    # Select a weighted number of negative edges 
    ntp = tp.size(1)
    ntn = neg.size(1)
    
    pscore = decode(z, tp)
    nscore = decode(z, neg)
    score = torch.cat([pscore, nscore]).numpy()

    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return auc, ap

def get_neg_adj(ei, nn):
    neg = ~(
        to_dense_adj(
            ei,
            max_num_nodes=nn
        ).bool()
    ).long()[0]

    neg = dense_to_sparse(neg)[0]
    return neg

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

def rw_sample(ei, num_nodes, wl=1):
    rw = random_walk(ei[0], ei[1], torch.arange(num_nodes), wl)
    return rw[:, 1:]
    
'''
Iteratively build out min distance matrix
Exp is used to smooth output scores 
(1/x is too steep, the x^(3/4) relationship is well known in edge distro so we 
choose that)
'''
def rw_distance(ei, batch=None, walk_len=5, dist_mat=None, 
                exp=0.75, num_nodes=None, window_size=None,
                num_walks=1):
    
    num_nodes = num_nodes if num_nodes else ei.max()
    if not window_size:
        window_size = walk_len
    
    if type(batch) == type(None):    
        batch = torch.tensor(list(range(num_nodes)))
    
    if type(dist_mat) == type(None):
        dist_mat = torch.zeros((num_nodes, num_nodes))
        dist_mat[ei[0], ei[1]] = 1

    rw = random_walk(ei[0], ei[1], batch.repeat(num_walks), walk_len)

    for i in range(2, window_size):
        score = 1 / (i ** exp)

        for j in range(rw.size(1)-i):
            # Find more efficient way to do this
            vals = dist_mat[rw[:, j], rw[:, j+i]]
            vals[vals < score] = score
            dist_mat[rw[:, j], rw[:, j+i]] = vals

    return dist_mat

'''
Splits edges into 85:5:10 train val test partition
(Following route of ARGAE paper)
'''
def edge_tvt_split(data):
    ne = data.edge_index.size(1)
    val = int(ne*0.85)
    te = int(ne*0.90)

    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)
    masks[0, rnd[:val]] = True 
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True 

    return masks[0], masks[1], masks[2]


'''
Uses Kipf-Welling pull #25 to quickly find negative edges
(For some reason, this works a touch better than the builtin 
torch geo method)
'''
def fast_negative_sampling(edge_list, batch_size, oversample=1.25):
    num_nodes = edge_list.max().item() + 1
    
    # For faster membership checking
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes

    el1d = el_hash(edge_list).numpy()
    neg = np.array([[],[]])

    while(neg.shape[1] < batch_size):
        maybe_neg = np.random.randint(0,num_nodes, (2, int(batch_size*oversample)))
        neg_hash = el_hash(maybe_neg)
        
        neg = np.concatenate(
            [neg, maybe_neg[:, ~np.in1d(neg_hash, el1d)]],
            axis=1
        )

    # May have gotten some extras
    neg = neg[:, :batch_size]
    return torch.tensor(neg).long()

'''
Given a list of src nodes, generate dst nodes that don't exist
'''
def in_order_negative_sampling(edge_list, src):
    # For faster membership checking
    num_nodes = edge_list.max().item()
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes
    el1d = el_hash(edge_list).numpy()

    src = src.numpy()
    dst = np.full((src.shape[0],), -1, dtype=np.long)
    while (dst.min() == -1):
        maybe_neg = np.random.randint(0, num_nodes+1, dst.shape)
        check = src + maybe_neg*num_nodes
        mask = (~np.in1d(check, el1d)) 
        dst[mask] = maybe_neg[mask]

    return torch.tensor(dst).long()

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