import torch 
import copy 

from torch import nn 
from torch_geometric.nn import GCNConv

class MyGAE(nn.Module):
    def __init__(self, feat_dim, hidden_dim=16, embed_dim=32):
        super(MyGAE, self).__init__()

        self.c1 = GCNConv(feat_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim)
        self.sig = nn.Sigmoid()

        self.embed_dim = embed_dim 
        self.loss = nn.MSELoss()

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        return self.c2(x, ei, edge_weight=ew)

    def recon(self, x, ei, ew=None):
        z = self.forward(x, ei, ew=ew)
        return self.sig(torch.mm(z,z.T))

    '''
    Calculate dot product of x1 and x2 assuming they're both
    "stacks of vectors" rather than full matrices on their own 

    NOTE: rather than a 0/1 probability of neighbors, we now want
    to reconstruct a matrix where 1 represents 1-hop, 1/2 represents 2-hop
    and so on (trending to 0)
    '''
    def loss_fn(self, x1, x2, target):
        x1 = x1.reshape(x1.size(0), 1, self.embed_dim)
        x2 = x2.reshape(x2.size(0), self.embed_dim, 1)

        learned_score = torch.sigmoid(torch.matmul(x1, x2)).squeeze(1)
        return self.loss(learned_score, target)

'''
This cant possibly be right...

Cora        AUC: 0.9435  AP: 0.9442
Citeseer:   AUC: 0.8993  AP: 0.8915 (yeah, see it wasn't right)

Compare to regular GAE on the same data

Cora:       AUC: 0.8990  AP: 0.8838 (0.910 & 0.920 are reported)
Citeseer:   AUC: 0.9160  AP: 0.9033 (0.895 & 0.899 are reported)

Using the Planetoid data set (which is pretty standard)

NOTE: on unbalanced data (ie, using 10x the neg samples than pos)
this performs semi-poorly, but by cranking up the negative sampling
rate, the AP gets back up to ~0.82 which seems.. okay (still better
than GAE which cant do that, as it merely makes the adj mat)
'''
def train_MyGAE(data, epochs=200, lr=0.001, K=10, SE_VAL=25,
                embed_dim=16, hidden_dim=32, neg_sampling=1):
    from .utils import get_score, rw_distance
    from torch.optim import Adam 
    from torch_geometric.utils import to_dense_adj, dense_to_sparse

    nn = data.x.size(0)
    ne = data.edge_index.size(1)

    G = MyGAE(data.x.size(1), embed_dim=embed_dim, hidden_dim=hidden_dim)
    
    rnd = torch.randperm(ne)
    g_opt = Adam(G.parameters(), lr=lr)

    # Bootstrapping score mat (it will be updated periodically later, however)
    scores = rw_distance(
        data.edge_index, 
        num_nodes=data.x.size(0),
        window_size=5,
        walk_len=K
    )

    # Partition data 
    rnd = torch.randperm(ne)
    tr, val, te = (
        rnd[:int(ne*0.85)], 
        rnd[int(ne*0.85):int(ne*0.90)], 
        rnd[int(ne*0.9):]
    )
    tv = torch.cat([tr,val])

    p1, p2 = scores.nonzero(as_tuple=True)
    score = scores[p1,p2]
    num_pos = p1.size(0)
    rnd = torch.randperm(num_pos)
    ptr, pva, pte = (
        rnd[:int(num_pos*0.85)], 
        rnd[int(num_pos*0.85):int(num_pos*0.90)], 
        rnd[int(num_pos*0.9):]
    )

    n1, n2 = (scores==0).nonzero(as_tuple=True)
    num_neg = n1.size(0)
    rnd = torch.randperm(num_neg)
    ntr, nva, nte = (
        rnd[:int(num_neg*0.85)], 
        rnd[int(num_neg*0.85):int(num_neg*0.90)], 
        rnd[int(num_neg*0.9):]
    )

    # Build adj mats to test overall loss 
    adj_tr = to_dense_adj(data.edge_index[:, tr], max_num_nodes=nn)[0]
    neg_val = ~(
        to_dense_adj(
            data.edge_index[:, tv],
            max_num_nodes = nn
        ).bool()
    ).long()[0]

    neg_val = dense_to_sparse(neg_val)[0]

    best = (0, None)
    stop_early = 0

    for e in range(epochs):
        # Train encoder 
        g_opt.zero_grad()
        x = G(data.x, data.edge_index[:, tr])

        # Presumably more negatives than positives
        neg_samples = ntr[torch.randperm(ntr.size(0))[:num_pos*neg_sampling]]

        pos_loss = G.loss_fn(
            x[p1[ptr]], 
            x[p2[ptr]], 
            score[ptr].unsqueeze(-1)
        )
        neg_loss = G.loss_fn(
            x[n1[neg_samples]],
            x[n2[neg_samples]], 
            torch.zeros(neg_samples.size(0), 1)
        )

        g_loss = pos_loss + neg_loss 
        g_loss.backward()
        g_opt.step()

        g_loss = g_loss.item()

        # Check AUC  
        auc, ap = get_score(
            data.edge_index[:, val], 
            neg_val, 
            G.recon(data.x, data.edge_index[:, tv]).detach()
        )

        print("[%d] Loss: %0.4f  |  AUC: %0.3f AP: %0.3f" %
            (
                e, g_loss, auc, ap
            )
        )

        # Early stopping to prevent over-fitting
        vscore = (auc+ap).mean()
        if vscore > best[0]:
            best = (vscore, copy.deepcopy(G)) 
            stop_early = 0
        else:
            stop_early += 1
            if stop_early == SE_VAL:
                print("Early stopping")
                break
    
    # Final score and cleanup
    del adj_tr
    del neg_val 

    G = best[1]

    neg = dense_to_sparse(
        ~(
            to_dense_adj(
                data.edge_index
            ).bool()
        ).long()[0]
    )[0]

    auc, ap = get_score(
        data.edge_index[:, te],
        neg, 
        G.recon(data.x, data.edge_index).detach()
    )

    print("Final AUC: %0.4f  AP: %0.4f" % (auc, ap))
    return G