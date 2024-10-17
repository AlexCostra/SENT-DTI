import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import pickle
import sys
import os
from utils.word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
sys.path.append('..')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom,explicit_H=False,use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])

def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

def get_pretrain_prot(prot2id, protein, max_len=1000):
    prot_idx = []
    for i in range(len(protein)-2):
        prot_idx.append(prot2id.get(protein[i:(i+3)], len(prot2id)+1))
    return prot_idx

def get_pretrain_smi(smi2id, smi, max_len=100):
    smi_idx = [smi2id.get(i, len(smi2id)+1) for i in smi]
    return smi_idx
'''
a=np.load('./DTI_DATA//data1//network_data//drug_protein.npy')
b=np.load('./DTI_DATA//data1//network_data//drug_protein_indepent.npy')
p=np.load('./DTI_DATA//data1//sequence_drug_feature//protein.npy')
print('hi',np.array_equal(a,b))
dti_o = b
dti_test = a
# print('dti_o',  np.array_equal(dti_test,dti_o))
print('dti_o', dti_test.shape, dti_o.shape)
train_positive_index = []
test_positive_index = []
whole_negative_index = []

for i in range(np.shape(dti_o)[0]):
    for j in range(np.shape(dti_o)[1]):
        if int(dti_o[i][j]) == 1:
            train_positive_index.append([i, j])

        elif int(dti_test[i][j]) == 1:
            test_positive_index.append([i, j])
        else:
            whole_negative_index.append([i, j])

negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                         size=len(test_positive_index) + len(train_positive_index),
                                         replace=False)
# f = open(f"{time.strftime('%m_%d_%H_%M_%S', time.localtime())}_negtive.txt", "w", encoding="utf-8")
# for i in negative_sample_index:
#     f.write(f"{i}\n")
# f.close()
data_set = np.zeros((len(negative_sample_index) + len(test_positive_index) + len(train_positive_index), 3),
                    dtype=int)
count = 0
train_index = []
test_index = []

for i in train_positive_index:
    data_set[count][0] = i[0]
    data_set[count][1] = i[1]
    data_set[count][2] = 1
    train_index.append(count)
    count += 1
print('count_train', count)
for i in test_positive_index:
    data_set[count][0] = i[0]
    data_set[count][1] = i[1]
    data_set[count][2] = 1
    test_index.append(count)
    count += 1
print('count_test', count)
f = open("./DTI_DATA//data1//network_data//dti_cledge.txt", "w", encoding="utf-8")
for i in range(count):
    for j in range(count):
        if data_set[i][0] == data_set[j][0] or data_set[i][1] == data_set[j][1]:
            f.write(f"{i}\t{j}\n")

for i in range(len(negative_sample_index)):
    data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
    data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
    data_set[count][2] = 0
    if i < len(train_positive_index):
        train_index.append(count)
    else:
        test_index.append(count)
    count += 1
f = open(f"./DTI_DATA//data1//network_data//dti_index.txt", "w", encoding="utf-8")
for i in data_set:
    f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

dateset = data_set
f = open("./DTI_DATA//data1//network_data//dtiedge.txt", "w", encoding="utf-8")
for i in range(dateset.shape[0]):
    for j in range(i, dateset.shape[0]):
        if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
            f.write(f"{i}\t{j}\n")

f.close()
'''



with open(f"./DTI_DATA//data1//sequence_drug_feature//independent_dataset.csv", "r") as f:
        data_list = f.read().strip().split('\n')

data_list = [d for d in data_list if '.' not in d.strip().split(',')[1]]
#print('fff',data_list[1])
N = len(data_list)

id2smi, smi2id, smi_embed = np.load('D:\学习\程序\\network comparasion\DTI_PROJECT\\smi2vec.npy',allow_pickle=True)
id2prot, prot2id, pro_embed = np.load('D:\学习\程序\\network comparasion\DTI_PROJECT\\prot2vec.npy',allow_pickle=True)
compounds, adjacencies,proteins,interactions,smi_ids,prot_ids = [], [], [], [], [], []
model = Word2Vec.load("D:\学习\程序\\network comparasion\DTI_PROJECT\\word2vec_30.model")
#print(smi2id)
for no, data in enumerate(data_list):
#print('/'.join(map(str, [no + 1, N])))
     smiles,sequence, interaction = data.strip().split(",")
     print(sequence)
     try:
            smi_id = get_pretrain_smi(smi2id, smiles)
            smi_ids.append(torch.LongTensor(smi_id))
            prot_id = get_pretrain_prot(prot2id, sequence)
            prot_ids.append(torch.LongTensor(prot_id))
            atom_feature, adj = mol_features(smiles)
            protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
            label = np.array(interaction, dtype=np.float32)
            atom_feature = torch.FloatTensor(atom_feature)
            adj = torch.FloatTensor(adj)
            protein = torch.FloatTensor(protein_embedding)
            label = torch.LongTensor(label)
            compounds.append(atom_feature)
            adjacencies.append(adj)
            proteins.append(protein)
            interactions.append(label)
     except:
            print('Error:', no)
            continue

dataset = list(zip(compounds, adjacencies, proteins, interactions, smi_ids, prot_ids))
#with open(f"../data/test.pickle", "wb") as f:#原始代码
with open(f"./DTI_DATA//data1//sequence_drug_feature//train.pickle", "wb") as f:
    pickle.dump(dataset, f)
print('The preprocess of dataset has finished!')
