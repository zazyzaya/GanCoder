import torch 

from torch import nn
from torch.autograd import Variable

'''
Using an adapted version of KD GAN 
https://arxiv.org/pdf/1711.04071.pdf

Which allows for indexing node representations the Discriminator/encoder
uses by taking the "realness" score the generated x and the real x produce
and using them as rewards s.t. backprop is possible over the probability
distro it uses

The generator is a simple NN that to output prob distro of nodes that are
convincing (but not real) neighbors of a given input node
'''
class RLGenerator(nn.Module):
    def __init__(self, num_nodes, embed_dim, hidden_dim=32):
        super(RLGenerator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, x):
        x = self.net(x)

        # Get indices to pass to discriminator
        probs = torch.softmax(Variable(x), dim=1)
        sample = torch.multinomial(probs, num_samples=1).squeeze()

        # Get log probability to pass to loss 
        log_probs = torch.log_softmax(x, dim=1)
        log_probs = log_probs[torch.arange(log_probs.size(0)), sample]
        return sample, log_probs

    '''
    Takes tensor of rewards (constants, w no attached grad)
    and back props r * log G(x' | x)
    '''
    def loss_fn(self, log_probs, rewards):
        loss = -torch.mean(Variable(torch.relu(rewards)) * log_probs)
        return loss 


'''
Not terribly important what this is, as long as its some sort of 
node autoencoder capible of finding X1X2^T approaching 1 if neighbors
0 else 

For this example, just using a GCN GAE to see if it's more successful than
a regular one (recycled from MyGAE code)
'''
from torch_geometric.nn import GCNConv
from .MyGAE import MyGAE
class RLDiscriminator(MyGAE):
    def __init__(self, feat_dim, embed_dim, hidden_dim=16, margin=0.01):
        super(RLDiscriminator, self).__init__(
            feat_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
        )

        self.margin=margin

    '''
    Also different, we need to calculate the marginal loss of the true sample
    and negative sample
    '''
    def score(self, x1, x2):
        x1 = x1.reshape(x1.size(0), 1, self.embed_dim)
        x2 = x2.reshape(x2.size(0), self.embed_dim, 1)

        return torch.sigmoid(torch.matmul(x1, x2)).squeeze(1)

    def loss_fn(self, t1, t2, f):
        pos_score = self.score(t1, t2)
        neg_score = self.score(t1, f)

        marg_loss = (1-pos_score) + neg_score
        return marg_loss

import copy 
from torch.optim import Adam 
from torch_cluster import random_walk
from torch_geometric.utils import to_dense_adj
from .utils import get_neg_adj, get_score, edge_tvt_split
def train_RLGAN(data, epochs=200, lr=0.001, SE_VAL=25, embed_dim=16, 
                hidden_dim=32, K=10):
    
    G = RLGenerator(data.x.size(0), embed_dim, hidden_dim=hidden_dim)
    D = RLDiscriminator(data.x.size(1), embed_dim, hidden_dim=hidden_dim)

    g_opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    edge_tvt_split(data)
    tr, te, va = (
        data.edge_tr,
        data.edge_te, 
        data.edge_va
    )
    tv = torch.logical_or(tr, va)
    tr_neg = get_neg_adj(data.edge_index[:, tr], data.num_nodes)
    val_neg = get_neg_adj(data.edge_index[:, tv], data.num_nodes)
    
    tr_adj = to_dense_adj(
        data.edge_index[:, tr],
        max_num_nodes=data.num_nodes
    )[0]

    batch = torch.arange(data.num_nodes)

    src = lambda mask : data.edge_index[:, mask][0, :]
    dst = lambda mask : data.edge_index[:, mask][1, :]
    rnd_pos = lambda mask : random_walk(src(mask), dst(mask), batch, 2)[:, 1].squeeze()
    rnd_neg = lambda : random_walk(tr_neg[0], tr_neg[1], batch, 2)[:, 1].squeeze()

    best = (0, None)
    se  = 0
    for e in range(epochs):
        d_opt.zero_grad()
        g_opt.zero_grad()
        
        # Get embeddings and samples
        x = D(data.x, data.edge_index[:, tr])
        gneg, log_prob = G(Variable(x))
        pos = rnd_pos(tr)
        tneg = rnd_neg()

        # Calculate Discriminator loss & update
        d_loss_f = D.loss_fn(x, x[pos], x[gneg]).mean()
        d_loss_t = D.loss_fn(x, x[pos], x[tneg]).mean() 
        d_loss = d_loss_f + d_loss_t
        d_loss.backward()
        d_opt.step()

        # Calculate Generator loss & update
        for _ in range(K):
            g_opt.zero_grad()
            neg, log_prob = G(Variable(x))
            x = D(data.x, data.edge_index[:, tr])

            reward = D.score(x, x[neg]).data
            reward[tr_adj[batch, neg] == 1] = 0 # Punish for using actual samples

            g_loss = G.loss_fn(log_prob, reward)
            g_loss.backward()
            g_opt.step()

        # Get validation data
        with torch.no_grad():
            x = D(data.x, data.edge_index[:, tr])
            adj = torch.matmul(x, x.T)

            auc, ap = get_score(
                data.edge_index[:, va], 
                val_neg, 
                adj
            )

        print(
            "[%d] D Loss: %0.4f  G Loss: %0.4f | Val: AUC %0.3f, AP %0.3f"
            % (e, d_loss.item(), g_loss.item(), auc, ap)
        )

        avg_score = (auc+ap)/2
        if avg_score > best[0]:
            best = (avg_score, copy.deepcopy(D))
            se = 0
        else:
            se += 1
            if se == SE_VAL:
                print("Early stopping!")
                break

    del val_neg
    neg = get_neg_adj(data.edge_index, data.num_nodes)
    D = best[1]

    with torch.no_grad():
        x = D(data.x, data.edge_index[:, tr])
        adj = torch.matmul(x, x.T)

        auc, ap = get_score(
            data.edge_index[:,te], 
            neg, 
            adj
        )

    print(
        "\nFinal score: AUC %0.4f, AP %0.4f"
        % (auc, ap)
    )