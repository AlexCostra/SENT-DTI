# SENT-DTI
The code is from our new paper in the field of Drug-Target prediction., entitled'' SENT-DTI: Semantic-Enhanced Drug-Target Interaction Prediction with Negative Training Strategy ''. SENT-DTI is a powerful tool designed for DTI prediction task, which leverages advanced algorithms to improve efficiency and accuracy.

## Folder Specification

- **main.py:** Please excuate this file.
- **Data** This file contain three types of data. i.e., proteinSquence.txt is a file that includes protein sequences, smiles.txt is a file of drug sequences and mat_drug_protein_drug.txt is a network file of Drug-Target network.
- **Utils:**  This file is responsible for processing sequence data including the generation of Drug-Protein Pair Network(DPPN) (generate_dataset_cli.py) , transformation of drug SMILE into molecule graph (data_preprocess.py), transformation of protein sequence into real-valued matrix (prot2vec.npy), transformation of drug SMILE into real-valued matrix (smi2vec.npy) and transformation of 3-g sequence into real-valued vvector with word2vec_30.model.
## Run the Code
### Data generation
(1) Please use proteinSquence.txt and smiles.txt to input into data_preprocess.py for precessing sequence data.
```bash
cd SENT-DTI/Utils
python data_preprocess .py '''

(2) Please use  mat_drug_protein_drug.txt  as input of generate_dataset_cli.py for generating DPP network.
```bash
cd SENT-DTI/Utils
python generate_dataset_cli.py

 
(3) To excecute SENT-DTI, please run the following command:

```bash
cd SENT-DTI
python main.py

## Acknowledgement
We sincerely thank Weiyu Shi for providing code. Please contact us with email: standyshi@qq.com