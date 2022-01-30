import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
from torch import optim
import matplotlib.pyplot as plt

def generate_a_graph(s, N):
    p = (12+s) / N
    q = (8-s) / N

    n = N // 2

    prob_thresh = np.zeros((N, N))
    prob_thresh[:n, :n] = p
    prob_thresh[n:, n:] = p
    prob_thresh[:n, n:] = q
    prob_thresh[n:, :n] = q
    W = np.random.rand(N, N)
    adj_mat = W < prob_thresh
    adj_mat = adj_mat * (np.ones(N) - np.eye(N))
    i_lower = np.tril_indices(N, -1)
    adj_mat[i_lower] = adj_mat.T[i_lower]
    labels = np.ones(N)
    labels[:n] = -1
    return torch.from_numpy(adj_mat), torch.from_numpy(labels)

def create_nx_from_adj(adj_mat):
    G = nx.Graph()
    n = len(adj_mat)
    G.add_nodes_from(list(range(n)))
    for u in range(0, n):
        for v in range(u+1, n):
            if adj_mat[u][v] == 1:
                G.add_edge(u, v)
    return G

def visualize_a_graph(G, s, plot=False):
    positions = nx.spring_layout(G)
    for i in range(0, G.order()):
        if i < G.order()//2:
            positions[i][0] += 3
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, node_size=8, width=0.1, pos=positions)
    ax.set_title(f'Community graph with s={s}')
    fig.tight_layout()
    if plot:
        plt.savefig(f'NetworkX-viz-s{s}.png')
        return
    plt.show()

def generate_vis():
    for s in range(1, 6):
        G, labels = generate_a_graph(s, 1000)
        visualize_a_graph(G, labels, s, True)

def spectral_cluster(adj_mat):
    w, v = np.linalg.eigh(adj_mat)
    s = np.argsort(w)
    return np.sign(v[:, s[-2]])

def overlap(pred, label):
    assert len(pred) == len(label)
    n = len(pred)
    try:
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
    except AttributeError:
        pass
    z1 = np.sum(np.sign(pred) == np.sign(label))
    z2 = np.sum(np.sign(pred) == np.sign(-label))
    return 2 * ((1 / n) * max(z1, z2) - 0.5)

class GNNModule(nn.Module):
    def __init__(self, n, alpha, beta):
        super(GNNModule, self).__init__()
        self.n = n
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.Tensor([alpha]).to(device), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([beta]).to(device), requires_grad=True)
        #self.norm = nn.BatchNorm1d(num_features=self.n)
        
    def forward(self, adj_mat, x):
        x1 = self.alpha * x
        x2 = x.repeat(self.n, 1) # n * n
        x2 = torch.sum(adj_mat * x, dim=1)
        x2 = self.beta * x2
        # x = self.norm(torch.add(x1, x2).view(1, 1, -1)).squeeze()
        x = self.relu(torch.add(x1, x2))
        return x

class GNN(nn.Module):
    def __init__(self, n, num_diffusion, alpha=0.8, beta=0.2):
        super(GNN, self).__init__()
        self.num_diffusion = num_diffusion
        self.module_list = nn.ModuleList([GNNModule(n, alpha, beta) for _ in range(num_diffusion)])
        self.norm = nn.InstanceNorm1d(num_features=n)
    def forward(self, adj_mat, x):
        for module in self.module_list:
            x = module(adj_mat, x)
            x = x.view(1, 1, -1)
            #print(x.dtype)
            #x = self.norm(x).squeeze()
            x = self.norm(x).squeeze()
        return x
    
    #def norm
            
def loss(pred, label, n=1000):
    l_z = torch.sum(torch.log(1+torch.exp(-pred*label)))
    l_z_invert = torch.sum(torch.log(1+torch.exp(pred*label)))
    return min(l_z, l_z_invert) / n

# Train the model with one graph
def train(gnn, adj_mat, labels, device):
    gnn.train()
    adj_mat = adj_mat.to(device)
    x = torch.sum(adj_mat, dim=1).to(device)
    labels = labels.to(device)
    y_hat = gnn(adj_mat, x)
    L = loss(y_hat, labels)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    return L

def test(gnn, s, n, device):
    gnn.eval()
    avg_overlap = 0
    for _ in range(100):
        adj_mat, labels = generate_a_graph(s, n)
        adj_mat = adj_mat.to(device)
        x = torch.sum(adj_mat, dim=1).to(device)
        labels = labels.to(device)
        pred = gnn(adj_mat, x)
        avg_overlap += overlap(pred, labels)
    print(f"Avg Overlap for 100 graphs: {avg_overlap / 100:.4f}")
    return avg_overlap / 100

def test_spectral_cluster(n):
    overlap_metric = []
    snr = []
    for s in range(1, 6):
        adj_mat, labels = generate_a_graph(s, n)
        pred = spectral_cluster(adj_mat)
        labels = labels.cpu().detach().numpy()
        overlap_metric.append(overlap(pred, labels))
        snr.append( ((12+s) - (8-s))**2 / (2 * 20) )
    plt.scatter(snr, overlap_metric)
    plt.xlabel("SNR")
    plt.ylabel("Average Overlap")
    for i in range(5):
        plt.annotate("s={}".format(i+1), (snr[i]+0.1, overlap_metric[i]+0.01))
    plt.savefig("Fiedler-Performance.png")


s = 5
n = 1000
K = 100
num_layers = 15
lr = 0.06

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Net = GNN(n, num_layers, 1, 1).to(device)
optimizer = optim.Adam(Net.parameters(), lr=lr)


cum_l = 0

snr = []
overlap_metric = []
for s in range(1, 6):
    print(f"Training GNN on community detection with s={s}, K={K} graphs, each with {n} nodes\nDiffusion Layers={num_layers}, lr={lr}")
    for i in range(K):
        adj_mat, labels = generate_a_graph(s, n)

        l = train(Net, adj_mat, labels, device)
        cum_l += l
        if (i+1) % 10 == 0:
            print(f"Epoch {i} Loss: {cum_l.item():.5f}")
            cum_l = 0
    snr.append( ((12+s) - (8-s))**2 / (2 * 20) )
    overlap_metric.append(test(Net, s, n, device))

# Generating performance graph
plt.scatter(snr, overlap_metric)
plt.xlabel("SNR")
plt.ylabel("Average Overlap")
for i in range(5):
    plt.annotate("s={}".format(i+1), (snr[i]+0.1, overlap_metric[i]+0.1))
plt.savefig("GCN-Performance.png")
print()

# Spectral Clustering Perfomance plot
#test_spectral_cluster(n)