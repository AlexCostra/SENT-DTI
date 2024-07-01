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
#from torch_geometric.nn import GATConv

from dgl.nn.pytorch import GraphConv,GINConv, GATConv
import random
from scipy import sparse
import copy
import networkx as nx
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
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sparse.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

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
    feature = torch.cat((h1[dateset[:, 1:2]], h2[dateset[:, 0:1]]), dim=-1)
    feature = feature.squeeze(1)
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
    return edge, feature



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
        support = torch.mm(input.double(), self.weight.double())
        #support=torch.nn.Tanh()(support)
        output = torch.spmm(adj.double(), support)
        #output=torch.nn.Tanh()(output)
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
        self.ln=nn.BatchNorm1d(hid_dim*2)

    def forward(self, x, adj):
        x = x.to(device)
        x=x.double()
        adj=adj.double()
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        #x1 = nn.PReLU()(self.gc1(x, adj).double())
        #x1=self.gc1(x, adj)
        #x1=nn.ELU()(self.gc1(x, adj))
        #x1=nn.Sigmoid()(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        #x2=nn.PReLU()(x2.double())
        res = x2
        #res=self.gc3(res,adj)
        return res
class CL_GCN(nn.Module):
    def __init__(self, nfeat, hid_dim,dropout,alpha = 0.8):
        super(CL_GCN, self).__init__()
        self.gcn1 = GCN(nfeat, hid_dim*2,dropout)
        #self.gcn1=GraphAttentionNetwork(nfeat, hid_dim*2,num_heads=3)
        self.alpha = alpha

    def forward(self, x1, adj1):
        z1 = self.gcn1(x1, adj1)
        return z1
class Predictor(nn.Module):
    def __init__(self, hid_dim,dropout,device,dticl):
        super(Predictor, self).__init__()
        self.device = device
        # protein encoding, target decoding
        self.out = nn.Sequential(
            nn.Linear(hid_dim * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.tau = 0.5
        self.CL_GCN=CL_GCN(128,hid_dim,dropout)
        self.cl=dticl
        self.trans_out=nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),

            #nn.Linear(hid_dim*2, 1024),
            #nn.ReLU(),
            #nn.Linear(1024,hid_dim*2),
            #nn.ReLU(),
        )
        self.trans_out1 = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),

            #nn.Linear(hid_dim * 2, 1024),
            #nn.ReLU(),
            #nn.Linear(1024, hid_dim * 2),
            #nn.ReLU(),
        )
        self.trans_out2 = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim*2),

            #nn.Linear(hid_dim * 2, 1024),
            #nn.ReLU(),
            #nn.Linear(1024, hid_dim * 2),
            #nn.ReLU(),
        )
        self.my_parameter = torch.randn(3, requires_grad=True)
        #self.gcn=network_representation_module_GCN(gcn_clm_node_dimension,gcn_hiddendim)
    def sim1(self, z1, z2,clm):
        z1=z1.double()
        z2=z2.double()
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        sim_matrix = sim_matrix.to(device)
        #loss = -torch.log(sim_matrix.sum(dim=-1)).mean()
        #loss=-torch.log(sim_matrix.mul(clm).sum(dim=-1)).mean()
        loss = -torch.log(sim_matrix.mul(clm).sum(dim=-1)).sum()
        return loss

    def sim(self, z1, z2):
        z1 = z1.double()
        z2 = z2.double()
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        sim_matrix = sim_matrix.to(device)
        #loss = -torch.log(sim_matrix.sum(dim=-1)).mean()
        loss = -torch.log(sim_matrix.sum(dim=-1)).sum()
        return loss

    def forward(self,  out_fc,feature, edge,train_indexs,test_indexs,test_indexs1,alpha=0.6):

        '''
        out_fc_drug_protein = self.trans_out(out_fc[:, 0:128])
        # out_fc_drug_protein = self.trans_out(torch.cat([out_fc[:,0:64], out_fc[:,256-64:256]], dim=-1))
        decoder_drug_protein = self.trans_out1(torch.cat([out_fc[:, 128:192], out_fc[:, 192:256]], dim=-1))
        interaction_feature_mol_protein = decoder_drug_protein
        # feature=out_fc_drug_protein
        feature = interaction_feature_mol_protein
        clf_interaction_features = self.CL_GCN(feature, edge)
        # clf_interaction_features=self.trans_out2(clf_interaction_features)
        # out_fc=torch.cat([self.my_parameter[0]*out_fc_drug_protein,self.my_parameter[1]*interaction_feature_mol_protein,self.my_parameter[2]*clf_interaction_features], dim=-1)
        # out_fc = torch.cat([out_fc_drug_protein, interaction_feature_mol_protein,clf_interaction_features], dim=-1)
        # out_fc = torch.cat([interaction_feature_mol_protein, clf_interaction_features], dim=-1)
        out_fc = torch.cat([out_fc_drug_protein, clf_interaction_features], dim=-1)
        interaction_feature_mol_protein = out_fc_drug_protein
        out_fc = self.out(out_fc.float())

        '''





        out_fc_drug_protein = self.trans_out(out_fc[:, 0:128])
        # out_fc_drug_protein = self.trans_out(torch.cat([out_fc[:,0:64], out_fc[:,256-64:256]], dim=-1))
        decoder_drug_protein = self.trans_out1(torch.cat([out_fc[:, 128:192], out_fc[:, 192:256]], dim=-1))
        interaction_feature_mol_protein = decoder_drug_protein
        # feature=out_fc_drug_protein
        feature = interaction_feature_mol_protein
        clf_interaction_features = self.CL_GCN(feature, edge)
        # clf_interaction_features=self.trans_out2(clf_interaction_features)
        # out_fc=torch.cat([self.my_parameter[0]*out_fc_drug_protein,self.my_parameter[1]*interaction_feature_mol_protein,self.my_parameter[2]*clf_interaction_features], dim=-1)
        # out_fc = torch.cat([out_fc_drug_protein, interaction_feature_mol_protein,clf_interaction_features], dim=-1)
        # out_fc = torch.cat([interaction_feature_mol_protein, clf_interaction_features], dim=-1)
        out_fc = torch.cat([out_fc_drug_protein, clf_interaction_features], dim=-1)############################
        #out_fc=clf_interaction_features
        #out_fc = out_fc_drug_protein
        interaction_feature_mol_protein = out_fc_drug_protein
        out_fc = self.out(out_fc.float())
        out_fc=torch.softmax(out_fc,1)

        '''
        interaction_feature_mol_protein = self.trans_out(out_fc[:,0:128])
        clf_interaction_features = self.CL_GCN(feature, edge)
        out_fc = self.out(out_fc[:,0:128].float())
        '''
        #interaction_feature_mol_protein=out_fc
        #clf_interaction_features=self.CL_GCN(feature, edge)
        #out_fc = self.out(clf_interaction_features.float())
        #out_fc = self.out(hhh.float())
        #interaction_feature_mol_protein=hhh[:,0:128]
        return interaction_feature_mol_protein,clf_interaction_features,out_fc[train_indexs],out_fc[test_indexs],out_fc[test_indexs1]

    def __call__(self, data,feature, edge,train_indexs,test_indexs,test_indexs1,correct_interaction,tjkst=True):
        Loss = nn.CrossEntropyLoss()
        interaction_feature_mol_protein,clf_interaction_features,predicted_interaction,test_results,test_results1= self.forward(data,feature, edge,train_indexs,test_indexs,test_indexs1)
        #centrol_point=torch.mean(predicted_interaction,0)
        #norm_=predicted_interaction-centrol_point
        #norm_square=torch.reciprocal(1+torch.norm(norm_, p=2))
        #norm_square=norm_square/torch.sum(norm_square)

        #loss1 = self.sim(interaction_feature_mol_protein, clf_interaction_features) + self.sim(clf_interaction_features, interaction_feature_mol_protein)
        #print('loss1',loss1)
        #loss1 = 0.2*self.sim1(interaction_feature_mol_protein, clf_interaction_features,self.cl) + 0.8*self.sim1(clf_interaction_features, interaction_feature_mol_protein,self.cl)
        loss1=0
        train_true = correct_interaction[train_indexs]
        '''
        train_posi_index=[]
        train_neg_index=[]
        count=0
        for i in train_true:
            if i==1:
                train_posi_index.append(count)
            else:
                train_neg_index.append(count)
            count=count+1
        '''
        #loss=-norm_square*torch.log(norm_square)
        if tjkst==True:

            loss = Loss(predicted_interaction, correct_interaction[train_indexs].reshape(-1).to(torch.long))
        else:
            loss = Loss(1-predicted_interaction, 1-correct_interaction[train_indexs].reshape(-1).to(torch.long))
        #loss_positive= labeled_loss = nn.BCEWithLogitsLoss()(outputs[is_labeled], labels[is_labeled])
        #unlabeled_loss = nn.BCEWithLogitsLoss()(outputs[~is_labeled], torch.zeros_like(labels[~is_labeled]))
        #loss=F.nll_loss(predicted_interaction, correct_interaction[train_indexs].reshape(-1).to(torch.long))
        #loss=0
        loss=0.8*loss+0.2*loss1
        correct_labels = correct_interaction[test_indexs].to('cpu').data.numpy()
        correct_labels1 = correct_interaction[test_indexs1].to('cpu').data.numpy()
        ys = test_results.to('cpu').data.numpy()
        ys1 = test_results1.to('cpu').data.numpy()
        predicted_labels = np.argmax(ys, axis=1)
        predicted_labels1 = np.argmax(ys1, axis=1)
        predicted_scores = ys[:, 1]
        predicted_scores1 = ys1[:, 1]
        cnmbbrwsj=np.argmax(predicted_interaction.to('cpu').data.numpy(), axis=1)
        return loss,train_true,predicted_interaction,correct_labels, predicted_labels, predicted_scores,cnmbbrwsj,correct_labels1, predicted_labels1,predicted_scores1

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









