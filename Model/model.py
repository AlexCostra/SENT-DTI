import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
import dgl
from dgl.nn.pytorch import GraphConv,GINConv, GATConv
import random
from scipy import sparse
import copy
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    print('fadj', fadj)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sparse.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    print('nfadj.shape', nfadj)

    return nfadj
def aug_random_edge(input_adj, drop_percent=0.4):

    percent = drop_percent

    edge_num = len(input_adj)  # 9228 / 2
    add_drop_num = int(edge_num * percent)
    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)
    drop_idx.sort()
    drop_idx.reverse()
    for i in drop_idx:
        input_adj = np.delete(input_adj, i, axis=0)
    return input_adj
def aug_random_mask(input_feature, drop_percent=0.4):
    # input_feature = input_feature.detach()
    input_feature = torch.tensor(input_feature)
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    mask_idx = random.sample(node_idx, mask_num)

    for i in range(input_feature.shape[0]):
        # mask_idx = random.sample(node_idx, mask_num)

        for j in mask_idx:
            aug_feature[i][j] = zeros
    return aug_feature

def constructure_graph(dateset, h1, h2, task="dti", aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=1)
    print('feature_shape',feature.shape)
    feature = feature.squeeze(1)
    print('feature_shape1', feature.shape)
    edge = np.loadtxt(f"./DTI_DATA//data1//network_data//dtiedge.txt", dtype=int)

    # for i in range(dateset.shape[0]):
    #     for j in range(i, dateset.shape[0]):
    #         if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
    #             edge.append([i, j])
    # fedge = np.array(generate_knn(feature.cpu().detach().numpy()))

    if aug:
        edge_aug = aug_random_edge(np.array(edge))
        edge_aug = load_graph(np.array(edge_aug), dateset.shape[0])
        edge = load_graph(np.array(edge), dateset.shape[0])

        feature_aug = aug_random_mask(feature)
        return edge, feature, edge_aug, feature_aug
    edge = load_graph(np.array(edge), dateset.shape[0])
    print('111',edge.shape)
    return edge, feature
class Mol_atoms_Attention_learning_module(nn.Module):
    def __init__(self, hid_dim, dropout, device, atom_dim=34):
        super(Mol_atoms_Attention_learning_module, self).__init__()
        self.dropout = dropout
        self.do = nn.Dropout(dropout)
        self.atom_dim = atom_dim

        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim)])
        self.W_gnn_trans = nn.Linear(atom_dim, hid_dim)
        self.device = device
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(len(self.W_gnn))])

    def forward(self, xs, A):
        for i in range(len(self.W_gnn)):
            h = torch.relu(self.W_gnn[i](xs))
            size = h.size()[0]
            N = h.size()[1]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1,
                                                                                                          2 * self.atom_dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(A > 0, e, zero_vec)  # 保证softmax 不为 0
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)
            xs = xs + h_prime
        xs = self.do(F.relu(self.W_gnn_trans(xs)))
        xs=xs.sum(dim=1)
        return xs
class Protein_features_learning_module(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5, 7], dropout_rate=0.5):
        super(Protein_features_learning_module, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2
        padding3 = (kernels[2] - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Linear(hid_dim*len(kernels), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
    def forward(self, protein):
        protein = protein.permute([0, 2, 1])  #[bs, hid_dim, seq_len]
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features3 = self.conv3(protein)
        features = torch.cat((features1, features2, features3), 1)  #[bs, hid_dim*3, seq_len]
        #features = features.max(dim=-1)[0]  #[bs, hid_dim*3]
        features=features.sum(dim=-1)#[bs, hid_dim*3]
        #features=features.mean(dim=-1)#[bs, hid_dim*3]
        return self.conv(features)

class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size , dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg

class Decoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super(Decoder, self).__init__()
        self.ft = nn.Linear(embed_dim, hid_dim)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)
        self.device = device

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.do(self.ft(trg))
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return trg

class network_representation_module_GCN(nn.Module):

    def __init__(self, in_feats, hidden_size):
        self.conv1 = nn.Sequential(
            GraphConv(in_feats, hidden_size, activation=None),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            GraphConv(hidden_size, hidden_size, activation=None),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
        )

    def forward(self, clm_all, inputs):
        h = self.conv1(clm_all, inputs)
        return h
class network_representation_module_GCN_residual(nn.Module):

    def __init__(self, in_feats, hidden_size,device):
            self.cov1=GraphConv(in_feats, hidden_size, activation=None)
            self.conv2_1=GraphConv(hidden_size, hidden_size, activation=None)
            self.conv2=GraphConv(hidden_size, hidden_size, activation=None)
            self.BN2=nn.BatchNorm1d(hidden_size)
            self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
            self.fc = nn.Linear(in_feats, hidden_size)
    def forward(self, clm_all, inputs):
        inputs=self.fc(inputs)
        h = self.conv2_1(clm_all, inputs)
        inputs=(inputs+h)*self.scale
        h = self.conv2(clm_all, inputs)
        inputs = (inputs + h) * self.scale
        inputs=self.BN2(inputs)
        return inputs
class network_representation_module_GIN(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(network_representation_module_GIN, self).__init__()
        self.conv1 = GINConv(in_feats, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = GINConv(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = self.bn1(x)  # Batch normalization
        x = torch.relu(x)
        x = self.conv2(g, x)
        x = self.bn2(x)  # Batch normalization
        x = torch.relu(x)
        return x
class network_representation_module_GIN_residual(nn.Module):

    def __init__(self, in_feats, hidden_size):
            self.cov1=GINConv(in_feats, hidden_size)
            self.conv2_1=GINConv(hidden_size, hidden_size)
            self.conv2=GINConv(hidden_size, hidden_size)
            self.BN2=nn.BatchNorm1d(hidden_size)
            self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
            self.fc = nn.Linear(in_feats, hidden_size)
    def forward(self, clm_all, inputs):
        inputs=self.fc(inputs)
        h = self.conv2_1(clm_all, inputs)
        inputs=(inputs+h)*self.scale
        h = self.conv2(clm_all, inputs)
        inputs = (inputs + h) * self.scale
        inputs=self.BN2(inputs)
        return inputs
class network_representation_module_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, num_heads):
        super(network_representation_module_GAT, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_size, num_heads=num_heads)
        self.conv2 = GATConv(hidden_size * num_heads, hidden_size, num_heads=num_heads)
    def forward(self, g, features):
        x = F.dropout(features, p=0.6, training=self.training)  # 输入特征进行Dropout
        x = self.conv1(g, x)
        x = F.elu(x)  # 使用ELU作为激活函数
        x = F.dropout(x, p=0.6, training=self.training)  # Dropout操作
        x = self.conv2(g, x)
        x = F.elu(x)
        return x
class network_representation_module_GAT_residual(nn.Module):

    def __init__(self, in_feats, hidden_size,num_heads):
            self.cov1=GATConv(in_feats, hidden_size, num_heads=num_heads)
            self.conv2_1=GATConv(hidden_size, hidden_size,num_heads=num_heads)
            self.conv2=GATConv(hidden_size, hidden_size,num_heads=num_heads)
            self.BN2=nn.BatchNorm1d(hidden_size)
            self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
            self.fc = nn.Linear(in_feats, hidden_size)
    def forward(self, clm_all, inputs):
        inputs=self.fc(inputs)
        h = self.conv2_1(clm_all, inputs)
        inputs=(inputs+h)*self.scale
        h = self.conv2(clm_all, inputs)
        inputs = (inputs + h) * self.scale
        inputs=self.BN2(inputs)
        return inputs
class GraphConvolution(Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GCN(nn.Module):
    def __init__(self, nfeat, hid_dim,dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, hid_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res
class CL_GCN(nn.Module):
    def __init__(self, nfeat, hid_dim,dropout,alpha = 0.8):
        super(CL_GCN, self).__init__()
        self.gcn1 = GCN(nfeat, hid_dim,dropout)
        self.alpha = alpha

    def forward(self, x1, adj1):
        z1 = self.gcn1(x1, adj1)
        return z1
class Predictor(nn.Module):
    def __init__(self, hid_dim, n_layers, kernel_size, n_heads, pf_dim, dropout, device, atom_dim=34,):
        super(Predictor, self).__init__()
        id2smi, smi2id, smi_embed = np.load('D:\学习\程序\\network comparasion\DTI_PROJECT\\smi2vec.npy',allow_pickle=True)
        id2prot, prot2id, prot_embed = np.load('D:\学习\程序\\network comparasion\DTI_PROJECT\\prot2vec.npy',allow_pickle=True)
        self.do = nn.Dropout(dropout)
        self.device = device
        self.prot_embed = nn.Embedding(len(prot_embed)+1, len(prot_embed[0]), padding_idx=0)
        self.prot_embed.data = prot_embed
        for param in self.prot_embed.parameters():
            param.requires_grad = False

        self.smi_embed = nn.Embedding(len(smi_embed)+1, len(smi_embed[0]), padding_idx=0)
        self.smi_embed.data = smi_embed
        for param in self.smi_embed.parameters():
            param.requires_grad = False
        print(f'prot Embed: {len(prot_embed)},  smi Embed: {len(smi_embed)}')

        # protein encoding, target decoding
        self.enc_prot = Encoder(len(prot_embed[0]), hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_smi = Decoder(len(smi_embed[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

        # target  encoding, protein decoding
        self.enc_smi = Encoder(len(smi_embed[0]), hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_prot = Decoder(len(prot_embed[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

        self.prot_textcnn =Protein_features_learning_module(100, hid_dim)
        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim)])
        self.W_gnn_trans = nn.Linear(atom_dim, hid_dim)
        self.out = nn.Sequential(
            nn.Linear(hid_dim * 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.CL_GCN=CL_GCN(128,hid_dim,dropout)
        self.dropout = dropout
        self.do = nn.Dropout(dropout)
        self.atom_dim = atom_dim
        #self.gcn=network_representation_module_GCN(gcn_clm_node_dimension,gcn_hiddendim)
        self.compound_attn = nn.ParameterList([nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(len(self.W_gnn))])
        self.gnn=Mol_atoms_Attention_learning_module( hid_dim, dropout, device, atom_dim=34)
    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask

        # compound = [bs,atom_num, atom_dim]
        # adj = [bs, atom_num, atom_num]
        # protein = [bs, protein_len, 100]
        # smi_ids = [bs, smi_len]
        # prot_ids = [bs, prot_len]

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        print('sim_matrix1', sim_matrix.shape)
        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        print('sim_matrix2', sim_matrix.shape)
        sim_matrix = sim_matrix.to(device)
        print('sim_matrix3', sim_matrix.shape)
        print('(torch.sum(sim_matrix, dim=1)', torch.sum(sim_matrix, dim=1).shape)
        loss = -torch.log(sim_matrix.sum(dim=-1)).mean()
        return loss

    def forward(self, compound, adj, protein, smi_ids, prot_ids, smi_num, prot_num,feature, edge,train_indexs,test_indexs,alpha=0.6):
        cmp_gnn_out = self.gnn(compound, adj)  # [bs, new_len, hid_dim]
        pro_textcnn_out = self.prot_textcnn(protein)  # [bs, prot_len, hid_dim]

        smi_max_len = smi_ids.shape[1]
        prot_max_len = prot_ids.shape[1]

        smi_mask, prot_mask = self.make_masks(smi_num, prot_num, smi_max_len, prot_max_len)
        out_enc_prot = self.enc_prot(self.prot_embed(prot_ids))  # [bs, prot_len, hid_dim]
        out_dec_smi = self.dec_smi(self.smi_embed(smi_ids), out_enc_prot, smi_mask, prot_mask)  # [bs, smi_len, hid_dim]

        prot_mask, smi_mask = self.make_masks(prot_num, smi_num, prot_max_len, smi_max_len)
        out_enc_smi = self.enc_smi(self.smi_embed(smi_ids))  # [bs, smi_len, hid_dim]
        out_dec_prot = self.dec_prot(self.prot_embed(prot_ids), out_enc_smi, prot_mask,
                                     smi_mask)  # # [bs, prot_len, hid_dim]

        # print(cmp_gnn_out.shape, pro_textcnn_out.shape, out_dec_smi.shape, out_dec_prot.shape)
        is_max = False
        if is_max:
            cmp_gnn_out = cmp_gnn_out.max(dim=1)[0]
            if pro_textcnn_out.ndim >= 3: pro_textcnn_out = pro_textcnn_out.max(dim=1)[0]
            out_dec_smi = out_dec_smi.max(dim=1)[0]
            out_dec_prot = out_dec_prot.max(dim=1)[0]
        else:
            cmp_gnn_out = cmp_gnn_out.mean(dim=1)
            if pro_textcnn_out.ndim >= 3: pro_textcnn_out = pro_textcnn_out.mean(dim=1)
            out_dec_smi = out_dec_smi.mean(dim=1)
            out_dec_prot = out_dec_prot.mean(dim=1)
        interaction_feature_mol_protein=torch.cat([out_dec_smi, out_dec_prot], dim=-1)
        clf_interaction_features = self.CL_GCN(feature, edge)
        #clf_interaction_features=self.gcn(clf,clf_features[train_indexs])
        out_fc = torch.cat([cmp_gnn_out, pro_textcnn_out, out_dec_smi, out_dec_prot, clf_interaction_features], dim=-1)
        out_fc=F.softmax(self.out(out_fc),dim=-1)
        return interaction_feature_mol_protein,clf_interaction_features,out_fc[train_indexs],out_fc[test_indexs]

    def __call__(self, data):
        compound, adj, protein, correct_interaction, smi_ids, prot_ids, atom_num, protein_num , \
        feature,edge,train_indexs,test_indexs= data
        Loss = nn.CrossEntropyLoss()
        interaction_feature_mol_protein,clf_interaction_features,predicted_interaction,test_results= self.forward(compound, adj, protein, smi_ids, prot_ids, atom_num, protein_num,feature, edge,train_indexs,test_indexs)
        alpha=0.6
        loss1 = alpha * self.sim(interaction_feature_mol_protein, clf_interaction_features) + (alpha) * self.sim(
        clf_interaction_features, interaction_feature_mol_protein)
        train_true = correct_interaction[train_indexs]
        loss = Loss(predicted_interaction, train_true)
        reg = get_L2reg(Predictor.parameters())
        loss=loss+loss1+1e-4*reg
        correct_labels = correct_interaction[test_indexs].to('cpu').data.numpy()
        ys = test_results.to('cpu').data.numpy()
        predicted_labels = np.argmax(ys, axis=1)
        predicted_scores = ys[:, 1]
        return loss,train_true,predicted_interaction,correct_labels, predicted_labels, predicted_scores

def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)
MAX_PROTEIN_LEN = 1500
MAX_DRUG_LEN = 200
def pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num, protein_num = [], []

    for atom in atoms:
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    for protein in proteins:
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    if atoms_len>MAX_DRUG_LEN: atoms_len = MAX_DRUG_LEN
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        if a_len>atoms_len: a_len = atoms_len
        atoms_new[i, :a_len, :] = atom[:a_len, :]
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        if a_len>atoms_len: a_len = atoms_len
        adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
        i += 1

    if proteins_len>MAX_PROTEIN_LEN: proteins_len = MAX_PROTEIN_LEN
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        if a_len>proteins_len: a_len = proteins_len
        proteins_new[i, :a_len, :] = protein[:a_len, :]
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    smi_id_len = 0
    for smi_id in smi_ids:
        atom_num.append(len(smi_id))
        if len(smi_id) >= smi_id_len:
            smi_id_len = len(smi_id)

    if smi_id_len>MAX_DRUG_LEN: smi_id_len = MAX_DRUG_LEN
    smi_ids_new = torch.zeros([N, smi_id_len], dtype=torch.long, device=device)
    for i, smi_id in enumerate(smi_ids):
        t_len = len(smi_id)
        if t_len>smi_id_len: t_len = smi_id_len
        smi_ids_new[i, :t_len] = smi_id[:t_len]
    ##########################################################
    prot_id_len = 0
    for prot_id in prot_ids:
        protein_num.append(len(prot_id))
        if len(prot_id) >= prot_id_len: prot_id_len = len(prot_id)

    if prot_id_len>MAX_PROTEIN_LEN: prot_id_len = MAX_PROTEIN_LEN
    prot_ids_new = torch.zeros([N, prot_id_len], dtype=torch.long, device=device)
    for i, prot_id in enumerate(prot_ids):
        t_len = len(prot_id)
        if t_len>prot_id_len: t_len = prot_id_len
        prot_ids_new[i, :t_len] = prot_id[:t_len]
    return (atoms_new, adjs_new, proteins_new, labels_new, smi_ids_new, prot_ids_new, atom_num, protein_num)

from transformers import AdamW, get_cosine_schedule_with_warmup




