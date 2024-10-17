import pickle
import torch
import numpy as np
import random
import os
import argparse
from model import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import auc as auc3
from model import constructure_graph
import tqdm
import timeit
os.chdir(os.path.dirname(os.path.abspath(__file__)))
lr = 0.0001
weight_decay = 1e-10
epochs=1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_cross(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        print('train_index',train_index)
        set2.append(test_index)
        print('test_index', test_index)
    return set1, set2


def get_roc(out, label):
    return roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy())


def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg
def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2
def get_cross(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        print('train_index',train_index)
        set2.append(test_index)
        print('test_index', test_index)
    return set1, set2

def init_seed(SEED = 2021):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

def train(model, data_pack, optim, train_index, test_index, epoch, fold):
            model.train()
            a = list(data_pack)
            a.append(train_index)
            a.append(test_index)
            data_pack = tuple(a)

            loss, label, out, test_label, predictive_test_label, predictive_out = model(data_pack)
            train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(dtype=float) / len(train_index)
            task1_roc = get_roc(out, label[train_index])
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print(f"{epoch} epoch loss  {loss:.4f} train is acc  {train_acc:.4f}, task1 roc is {task1_roc:.4f},")
            te_acc, te_task1_roc1, te_task1_pr = main_test(predictive_out, predictive_test_label, test_index, epoch,
                                                           fold)

            return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr

def main_test(out, label, test_index, epoch, fold):
            acc1 = (out.argmax(dim=1) == label.reshape(-1)).sum(dtype=float) / len(test_index)

            task_roc = get_roc(out, label)

            task_pr = get_pr(out, label)
            # if epoch == 999:
            #     f = open(f"{fold}out.txt","w",encoding="utf-8")
            #     for o in  (out.argmax(dim=1) == label[test_index].reshape(-1)):
            #         f.write(f"{o}\n")
            #     f.close()
            return acc1, task_roc, task_pr


def pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, feature, edge):
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

        if atoms_len > MAX_DRUG_LEN: atoms_len = MAX_DRUG_LEN
        atoms_new = torch.zeros((N, atoms_len, 34), device=device)
        i = 0
        for atom in atoms:
            a_len = atom.shape[0]
            if a_len > atoms_len: a_len = atoms_len
            atoms_new[i, :a_len, :] = atom[:a_len, :]
            i += 1
        adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
        i = 0
        for adj in adjs:
            a_len = adj.shape[0]
            adj = adj + torch.eye(a_len)
            if a_len > atoms_len: a_len = atoms_len
            adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
            i += 1

        if proteins_len > MAX_PROTEIN_LEN: proteins_len = MAX_PROTEIN_LEN
        proteins_new = torch.zeros((N, proteins_len, 100), device=device)
        i = 0
        for protein in proteins:
            a_len = protein.shape[0]
            if a_len > proteins_len: a_len = proteins_len
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

        if smi_id_len > MAX_DRUG_LEN: smi_id_len = MAX_DRUG_LEN
        smi_ids_new = torch.zeros([N, smi_id_len], dtype=torch.long, device=device)
        for i, smi_id in enumerate(smi_ids):
            t_len = len(smi_id)
            if t_len > smi_id_len: t_len = smi_id_len
            smi_ids_new[i, :t_len] = smi_id[:t_len]
        ##########################################################
        prot_id_len = 0
        for prot_id in prot_ids:
            protein_num.append(len(prot_id))
            if len(prot_id) >= prot_id_len: prot_id_len = len(prot_id)

        if prot_id_len > MAX_PROTEIN_LEN: prot_id_len = MAX_PROTEIN_LEN
        prot_ids_new = torch.zeros([N, prot_id_len], dtype=torch.long, device=device)
        for i, prot_id in enumerate(prot_ids):
            t_len = len(prot_id)
            if t_len > prot_id_len: t_len = prot_id_len
            prot_ids_new[i, :t_len] = prot_id[:t_len]
        return (atoms_new, adjs_new, proteins_new, labels_new, smi_ids_new, prot_ids_new, atom_num, protein_num,feature, edge)



from sklearn.model_selection import StratifiedKFold
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='independent-test')
    parser.add_argument('--model_name', type=str, default='independent-test', help='The name of models')
    parser.add_argument('--protein_dim', type=int, default=100, help='embedding dimension of proteins')
    parser.add_argument('--atom_dim', type=int, default=34, help='embedding dimension of atoms')
    parser.add_argument('--hid_dim', type=int, default=64, help='embedding dimension of hidden layers')
    parser.add_argument('--n_layers', type=int, default=3, help='layer count of networks')
    parser.add_argument('--n_heads', type=int, default=8, help='the head count of self-attention')
    parser.add_argument('--pf_dim', type=int, default=256, help='dimension of feedforward neural network')
    parser.add_argument('--dropout', type=float, default=0.2, help='the ratio of Dropout')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--iteration', type=int, default=100, help='the iteration for training')
    parser.add_argument('--n_folds', type=int, default=5, help='the fold count for cross-entropy')
    parser.add_argument('--seed', type=int, default=2021, help='the random seed')
    parser.add_argument('--kernel_size', type=int, default=9, help='the kernel size of Conv1D in transformer')
    args = parser.parse_args()
    in_size=128

    with open('./DTI_DATA//data1//sequence_drug_feature//train.pickle',"rb") as f:
        data = pickle.load(f)
    #data_train = shuffle_dataset(data, 1234)
    data_train=data
    num=[708,1512]
    dtidata=np.loadtxt("./DTI_DATA/data1/network_data/dti_index.txt", dtype=int)
    print(dtidata)
    random_smiles_feature=torch.randn((num[0], in_size))
    random_protein_feature = torch.randn((num[1], in_size))
    features_d = random_smiles_feature.to(device)
    features_p = random_protein_feature.to(device)
    edge, feature = constructure_graph(dtidata, features_d, features_p)


    def main(tr, te, seed):
        all_acc = []
        all_roc = []
        all_f1 = []
        for i in range(len(tr)):
            f = open(f"{i}foldtrain.txt", "w", encoding="utf-8")
            train_index = tr[i]
            print('train_index', len(train_index))
            for train_index_one in train_index:
                f.write(f"{train_index_one}\n")
            test_index = te[i]
            print('test_index', len(test_index))
            f = open(f"{i}foldtest.txt", "w", encoding="utf-8")
            for train_index_one in test_index:
                f.write(f"{train_index_one}\n")
            #
            # if not os.path.isdir(f"{dir}"):
            #     os.makedirs(f"{dir}")

            model = Predictor(args.hid_dim, args.n_layers, args.kernel_size, args.n_heads, args.pf_dim,
                              args.dropout, device, atom_dim=34
            ).to(device)
            # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
            optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
            best_acc = 0
            best_f1 = 0
            best_roc = 0
            epochs=1000
            for epoch in range(epochs):
                loss, train_acc, task1_roc, acc, task1_roc1, task1_pr = train(model, data_pack,optim, train_index, test_index,
                                                                              epoch, i)
                if acc > best_acc:
                    best_acc = acc
                if task1_pr > best_f1:
                    best_f1 = task1_pr
                if task1_roc1 > best_roc:
                    best_roc = task1_roc1
                    # torch.save(obj=model.state_dict(), f=f"{dir}/net.pth")
            all_acc.append(best_acc)
            all_roc.append(best_roc)
            all_f1.append(best_f1)
            print(f"fold{i}  auroc is {best_roc:.4f} aupr is {best_f1:.4f} ")

        print(
            f"{'Shi'},{sum(all_acc) / len(all_acc):.4f},  {sum(all_roc) / len(all_roc):.4f} ,{sum(all_f1) / len(all_f1):.4f}")


    adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []
    for data in data_train:
        atom, adj, protein, label, smi_id, prot_id = data
        adjs.append(adj)
        atoms.append(atom)
        proteins.append(protein)
        labels.append(label)
        smi_ids.append(smi_id)
        prot_ids.append(prot_id)
    data_pack = pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, feature, edge)
    tr, te = get_cross(dtidata)
    main(tr, te,1000)



