import torch 

from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_cluster import random_walk


class DiscriminatorSingleAdj(nn.Module):
    def __init__(self, data, out_dim, hidden=32, num_convs=3, final_nn=False):
        super(DiscriminatorSingleAdj, self).__init__()

        self.in_dim = data.x.size()[1]
        self.out_dim = out_dim 
        self.num_nodes = data.num_nodes

        row, col = data.edge_index
        self.adj = SparseTensor(
            row=row, col=col, 
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )

        layers = [GCNConv(self.in_dim, hidden), nn.ReLU()]
        for _ in range(num_convs-2):
            layers.append(GCNConv(hidden, hidden))
            layers.append(nn.ReLU())
        
        self.convs = nn.ModuleList(
            layers + [
            GCNConv(hidden, out_dim),
            nn.Tanh()
        ])

        if final_nn:
            self.out = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.Tanh()
            )

        else:
            self.out = lambda x : x

    def forward(self, data):
        x = data.x
        for i, layer in enumerate(self.convs):
            # Nonlinearities
            if i % 2:
                x = layer(x)
            # G Convs
            else:
                x = layer(x, data.edge_index, edge_weight=data.edge_weight)

        out = self.out(x)

        # Note that XX^T[n, m] is X_n * X_m^T so this returns a square 
        # matrix of link probabilities 
        # For really big datasets this may be infeasible
        # return torch.sigmoid(torch.mm(out, out.transpose(0,1)))
        
        return out

    '''
    Generates random walks through the graph using the generated adj mat
    if minibatch not false, should be numeric for how large the minibatch
    is. Starting nodes will be selected randomly
    '''
    def positive_samples(self, edge_index, n_samples=1, walk_len=4, minibatch=False):
        if minibatch:
            batch = torch.randperm(self.num_nodes)[minibatch]
        else:
            batch = torch.arange(0, self.num_nodes)

        batch = batch.repeat(n_samples)
        row, col = edge_index
        rw = random_walk(row, col, batch, walk_len)

        return rw 

'''
TODO change this to output random walk node ids, building a full
adj matrix is gonna be waaaaay too big
'''
class GeneratorSingleAdj(nn.Module):
    def __init__(self, num_nodes, latent_dim=16, hidden=128, n_layers=3):
        super(GeneratorSingleAdj, self).__init__()

        self.num_nodes = num_nodes
        self.latent_dim = latent_dim

        layers = [nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2)]

        for i in range(1, n_layers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Linear(hidden, num_nodes))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(
            *layers
        )

    def forward(self, batch_size=None):
        if type(batch_size) == type(None):
            batch_size = self.num_nodes

        x = Variable(torch.empty((batch_size, self.latent_dim)).normal_(mean=0, std=1))
        x = self.layers(x)
        x = torch.round(x)

        return x.view((self.num_nodes, self.num_nodes))


'''
Returns probability that each random walk could have occurred in the given 
graph. Ideally, for true samples this should tend toward 1, and false samples
should tend toward zero
'''
def rw_link_score_sg(rw, x):
    trans_probabilities = []
    for step in range(rw.size()[1] - 1):
        trans_probabilities.append(
            x[rw[:, step], rw[:, step+1]].unsqueeze(-1)
        )

    # I'm pretty sure cat doesn't affect autograd
    # tree, but I guess we'll find out
    ret = torch.cat(trans_probabilities, dim=1)
    ret = ret.prod(1, keepdim=True)

    return ret

'''
Returns overall prob of node 1 cooccuring with other nodes in walk
'''
def rw_link_score_cbow(rw, x):
    start, rest = rw[:, 0], rw[:, 1:].contiguous()

    h_start = x[start].view(rw.size(0), 1, x.size(1))
    h_rest  = x[rest.view(-1)].view(rw.size(0), -1, x.size(1))

    out = (h_start * h_rest).sum(dim=-1).view(-1)
    return out

'''
If using skip gram, make sure to have discriminator return 
single step probabilites (XX^T) rather than embeddings
'''
def rw_link_score(rw, x, cbow=True):
    if cbow:
        return rw_link_score_cbow(rw, x)
    return rw_link_score_sg(rw, x)


# Test to make sure grad is attached to everything, 
# simulates one step of the epoch
if __name__ == '__main__':
    import time 
    from load_graphs import load_cora
    from torch.optim import Adam 
    
    print("Initializing...")
    data = load_cora()

    loss = nn.BCELoss()
    real_nodes = Variable(torch.full((data.num_nodes, 1), 1.0))
    fake_nodes = Variable(torch.zeros((data.num_nodes, 1)))

    D = DiscriminatorSingleAdj(data, 128)
    G = GeneratorSingleAdj(data.num_nodes)

    d_opt = Adam(D.parameters())
    g_opt = Adam(G.parameters())

    print("Starting...")
    start = time.time()
    
    # Train discriminator 
    d_opt.zero_grad()
    x = D(data)

    rw = D.positive_samples(data.edge_index)
    walk_scores = rw_link_score(rw, x)
    d_loss_r = loss(walk_scores, real_nodes)

    fake_adj = G()
    x = D(data)
    walk_scores_f = G.compare_to_disc(fake_adj, x)
    d_loss_f = loss(walk_scores_f, fake_nodes)
    
    d_loss_f.backward()
    d_loss_r.backward()
    d_opt.step()

    d_tot_loss = d_loss_f.mean() + d_loss_r.mean()

    print('Discriminator cycle: %0.2fs' % (time.time()-start))

    # Train generator
    gstart = time.time()
    
    g_opt.zero_grad()
    x = D(data)
    fake_adj = G()

    walk_scores_f = G.compare_to_disc(fake_adj, x)
    g_loss = loss(walk_scores_f, real_nodes)
    g_loss.backward()
    g_opt.step()

    g_tot_loss = g_loss.mean()

    print('Generator cycle: %0.2fs' % (time.time()-gstart))
    print('[0]: D Loss %0.5f\t G Loss %0.5f\t %0.2fs' % 
        (d_tot_loss.item(), g_tot_loss.item(), time.time()-start)
    )