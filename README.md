


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

