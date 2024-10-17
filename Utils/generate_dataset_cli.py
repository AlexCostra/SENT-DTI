import pandas as pd
import numpy as np
dataset =np.loadtxt("./DTI_DATA/data1/network_data/dti_index.txt", dtype=int)
print(dataset)
protein_sets=[]
with open(f"./DTI_DATA//data1//sequence_drug_feature//proteinSquence.txt", "r") as f:
    data_list = f.read().strip().split('\n')
for no, data in enumerate(data_list):
       if no%2 !=0:
        #print('/'.join(map(str, [no + 1, N])))
        m=data.strip().split(",")
        protein_sets.append(m)
smiles_sets=[]
with open(f"./DTI_DATA//data1//sequence_drug_feature//smlies.txt", "r") as f:
       data_smiles_list = f.read().strip().split('\n')
for no, data in enumerate(data_smiles_list):
        m = data.strip().split(",")
        print(m)
        smiles_sets.append(m)
all_mol_protein_annotation_samples=[]
for i in dataset:
    x=[]
    x.extend(smiles_sets[i[0]])
    x.extend(protein_sets[i[1]])
    x.extend([int(i[2])])
    all_mol_protein_annotation_samples.append(x)
data=pd.DataFrame(np.array(all_mol_protein_annotation_samples))
print(data)
data.to_csv('./DTI_DATA//data1//sequence_drug_feature//independent_dataset.csv',index=False)