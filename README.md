<h1 align="center">SENT-DTI: a computational method for prediction of Drug-Target Interactions</h1>

The code is from our new paper in the field of Drug-Target prediction., entitled'' SENT-DTI: Semantic-Enhanced Drug-Target Interaction Prediction with Negative Training Strategy ''. SENT-DTI is a powerful tool designed for DTI prediction task, which leverages advanced algorithms to improve efficiency and accuracy.
## Highlight
-  As for the lack of semantic interactive information on Drug Protein Pair Network (DPPN), we design a novel feature fuse method on DPPN. Specifically, It extracts Drug Protein Pair (DPP) interactive semantic features through a bidirectional encoder-decoder based on self-attention. These features can be further incorporated into DPPN and used to learn augmented DPP feature representation through a semantic-aware GCN on the semantic DPPN.
 -  We design a simple yet effective Negative Training (NT) strategy for false-negative drug-target association issue to adaptively optimize the probability distribution of positive DPPs through a binary association-specific negative loss function and identify false-negative DPP associations by a Unified High-Confidence false-negative DPP association filtering (UHCF) mechanism. To our knowledge, this is the first work to mitigate DPP false-negative association for the DTI prediction task.

 -  Our experimental results show that SENT-DTI outperforms SOTA baselines on benchmark
datasets, whereby ablation experiments further confirm the effectiveness of our two new techniques, i.e. Semantic DPPN representation learning and FNDNT.

## Model Architecture
A novel semantic-enhanced DPP representation learning method with a negative training strategy entitled SENT-DTI is proposed to identify potential DTIs. As depicted in the below figure, SENT-DTI comprises three main modules: interactive feature augmentation module, independent feature extraction module, and false-negative DPP association identification module.

![image](https://github.com/AlexCostra/SENT-DTI/blob/main/Utils/model_picture.png)

## Requirements
- Python >= 3.6
- Pytorch >= 1.7

## Folder Specification

- **main.py:** Please excuate this file.
- **Model:** It includes our model (model.py).
- **Data** This file contain three types of data. i.e., proteinSquence.txt is a file that includes protein sequences, smiles.txt is a file of drug sequences and mat_drug_protein_drug.txt is a network file of Drug-Target network.
- **Utils:**  This file is responsible for processing sequence data including the generation of Drug-Protein Pair Network(DPPN) (generate_dataset_cli.py) , transformation of drug SMILE into molecule graph (data_preprocess.py), transformation of protein sequence into real-valued matrix (prot2vec.npy), transformation of drug SMILE into real-valued matrix (smi2vec.npy) and transformation of 3-g sequence into real-valued vector with word2vec_30.model.
## Run the Code
#### (1) Sequence Preprocessing
  Please use proteinSquence.txt and smiles.txt to input into data_preprocess.py for precessing sequence data.
```bash
cd SENT-DTI/Utils
python data_preprocess .py 
``` 
#### (2) DPP Network generation
  Please use  mat_drug_protein_drug.txt  as input of generate_dataset_cli.py for generating DPP network.
```bash
cd SENT-DTI/Utils
python generate_dataset_cli.py
``` 
#### (3) DTI prediction
  To excecute SENT-DTI, please run the following command. The contents include model train and model test:

```bash
cd SENT-DTI
python main.py
``` 
## Acknowledgement
We sincerely thank Weiyu Shi for providing code. Please contact us with Email: standyshi@qq.com
<u><p><b><i><font size="6">If you are interested in Natural Language Processing and Large Language Models, feel free to contact us by Email: zhangyijia@dlmu.edu.cn </font></i></b></p>

