import pickle

import pandas as pd
import torch
import numpy as np
import random
import os
import argparse
from sent import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import auc as auc3
from sent import constructure_graph
from tqdm import tqdm
import timeit
import warnings
cnm=0.9

epochs1111111111111111=51
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
lr = 0.0001
weight_decay = 1e-10
# epochs=100#1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_clGraph(data, task):
    #cledg = np.loadtxt(f"./DTI_DATA/data1/network_data/{task}_cledge.txt", dtype=int)
    #cledg = np.loadtxt(f'D:\学习\程序\\network comparasion\DTI_PROJECT\DTI_DATA\datat2\\dti_IFDTI_cledge.txt',dtype=int)
    #cledg = np.loadtxt(f"D:\学习\程序\\network comparasion\PMF-CPI-main\PMF-CPI-main\datasets\CYP_cls\\train/dti_CYP_PMF-CPI_cledge.txt", dtype=int)
    #cledg = np.loadtxt(f"./DTI_DATA/data1/network_data/{task}_Icledge.txt", dtype=int)
    #cledg = np.loadtxt(f"D:\学习\程序\\network comparasion\PMF-CPI-main\PMF-CPI-main\datasets\BindingDB_cls\ind_test\dti_BindingDB_PMF-CPI_cledge.txt",dtype=int)
    cledg = np.loadtxt(f'D:\学习\程序\\network comparasion\\transformerCPI-master\\transformerCPI-master\data\GPCR_train\\dti_gpcr_train_cledge.txt', dtype=int)
    cl = torch.eye(len(data))
    for i in cledg:
        cl[i[0]][i[1]] = 1
    return cl


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
        set2.append(test_index)
    return set1, set2


def get_roc(out, label):
    label = torch.tensor(label)
    out = torch.tensor(out)
    return roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy())
    # return roc_auc_score(label.cpu(), out.cpu().detach().numpy())


def get_roc1(out, label):
    label = torch.tensor(label)
    out = torch.tensor(out)
    return roc_auc_score(label.cpu(), out.cpu().detach().numpy())


def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)


def get_pr1(out, label):
    label = torch.tensor(label)
    out = torch.tensor(out)
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out.cpu().detach().numpy())
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
        set2.append(test_index)
    return set1, set2


def init_seed(SEED=2021):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    default_configure = {
        'batch_size': 20
    }


    def set_random_seed(seed=0):
        """Set random seed.
        Parameters
        ----------
        seed : int
            Random seed to use
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


    def setup(args, seed):
        args.update(default_configure)
        set_random_seed(seed)
        return args


    seed = 47
    args = setup(default_configure, seed)
    s = 47
    in_size = 64
    hidden_size = 64
    out_size = 128
    dropout = 0.5
    lr = 0.0001
    weight_decay = 1e-10
    epochs = 200  # 原始参数1000
    epochs1=500
    cl_loss_co = 1
    reg_loss_co = 0.0001
    fold = 0
    dir = "../modelSave"

    args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
    for name in ["zheng"]:
        # for name in ["heter","Es","GPCRs","ICs","Ns","zheng"]:
        #dtidata = np.loadtxt("./DTI_DATA/data1/network_data/dti_index.txt", dtype=int)
        #dtidata = np.loadtxt("D:\学习\程序\\network comparasion\DTI_PROJECT\DTI_DATA\datat2\\IFDTI_dti_index.txt", dtype=int)
        #dtidata = np.loadtxt("D:\学习\程序\\network comparasion\PMF-CPI-main\PMF-CPI-main\datasets\CYP_cls\\train\\CYP_PMF-CPI_dti_index.txt", dtype=int)
        #dtidata = np.loadtxt("D:\学习\程序\\network comparasion\PMF-CPI-main\PMF-CPI-main\datasets\BindingDB_cls\ind_test\BindingDB_PMF-CPI_dti_index.txt",dtype=int)
        dtidata = np.loadtxt('D:\学习\程序\\network comparasion\\transformerCPI-master\\transformerCPI-master\data\GPCR_train\\gpcr_train_dti_index.txt',dtype=int)

        #num=[1514,708]
        #num = [1471,1606]
        #num=[4335,3]
        #num = [5017, 467]
        num = [5368, 347]
        dti_cl = get_clGraph(dtidata, "dti").to(args['device'])
        # dataName heter Es GPCRs ICs Ns zheng
        dti_label = torch.tensor(dtidata[:, 2:3]).to(args['device'])
        #data1 = pd.read_csv('./DTI_DATA/data1/sequence_drug_feature/interaction_feature_0.csv')
        #data1 = pd.read_csv("D:\学习\程序\\network comparasion\DTI_PROJECT\DTI_DATA\datat2\interaction_ifidti_modify_feature_0.csv")
        #data1 = pd.read_csv('D:\学习\程序\\network comparasion\PMF-CPI-main\PMF-CPI-main\datasets\CYP_cls\\train\\interaction_feature_CYP_PMF-CPI_1.csv')
        #data1=pd.read_csv('D:\学习\程序\\network comparasion\PMF-CPI-main\PMF-CPI-main\datasets\BindingDB_cls\ind_test\interaction_feature_BindingDB_PMF-CPI_0.csv')
        data1 = pd.read_csv('D:\学习\程序\\network comparasion\\transformerCPI-master\\transformerCPI-master\data\GPCR_train\\interaction_feature_gpcr_0.csv')

        print(data1)
        data1 = np.array(data1)
        data1 = torch.tensor(data1)
        # hd=data1[:,0:64]
        # hp=data1[:,64:128]
        hp = torch.randn((num[1], in_size))
        hd = torch.randn((num[0], in_size))
        features_d = hd.to(args['device'])
        features_p = hp.to(args['device'])
        edge, feature = constructure_graph(dtidata, features_d, features_p)
        data = dtidata
        label = dti_label
        is_ce=True

        def main(tr, te, seed):
            all_acc = []
            all_roc = []
            all_f1 = []
            all_acc1 = []
            all_roc1 = []
            all_f11 = []
            all_recall = []
            all_sens = []
            all_pre = []
            for i in range(len(tr)):
                if i == len(tr) - 1:
                    continue
                f = open(f"{i}foldtrain.txt", "w", encoding="utf-8")
                train_index = tr[i]
                for train_index_one in train_index:
                    f.write(f"{train_index_one}\n")
                test_index = te[i]
                f = open(f"{i}foldtest.txt", "w", encoding="utf-8")
                for train_index_one in test_index:
                    f.write(f"{train_index_one}\n")
                #
                # if not os.path.isdir(f"{dir}"):
                #     os.makedirs(f"{dir}")

                model = Predictor(hidden_size, dropout, device
                                  , dti_cl).to(args['device'])
                # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
                optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
                best_acc = 0
                best_f1 = 0
                best_roc = 0
                best_acc1 = 0
                best_f11 = 0
                best_roc1 = 0
                best_recall = 0
                best_pre = 0
                best_sens = 0
                for epoch in range(epochs1111111111111111):#fold1  auroc is 0.9316 aupr is 0.9256 accuracy is 0.8742 recall is 0.8626,precision is 0.8555,sensitivity is 0.8545 ,zheng,0.9316 ,0.9256,0.8742
                    # loss, train_acc, task1_roc, acc, task1_roc1, task1_pr,out1,labelcnm,p_y,p_yy,zzjj,dasbb = train(model, optim, data1,feature, edge,train_index, test_index, label)
                    # loss, train_acc, task1_roc, acc, task1_roc1, task1_pr ,recall, pre, sensitive= train(model, optim, data1, feature, edge,train_index, test_index, label)
                    loss, train_acc, task1_roc, acc1, task1_roc11, task1_pr1, out1, labelcnm, p_y, p_yy, zzjj, dasbb, acc, task1_roc1, task1_pr, labelcnmb, out1b, p_yb = train(
                        model, optim, data1, feature, edge, train_index, train_index, test_index, label,is_ce)
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
                # all_pre.append(best_pre)
                # all_sens.append(best_sens)
                # all_recall.append(best_recall)
                # print(f"fold{i}  auroc is {best_roc:.4f} aupr is {best_f1:.4f} accuracy is {best_acc:.4f} recall is {best_recall:4f} precision is {best_pre:4f} sensitivity is {best_sens:4f}")
                predictive1_count = [i for i in range(len(out1b)) if out1b[i] == 1]
                wc = labelcnmb.reshape(-1)
                actual_num = [i for i in range(len(wc)) if wc[i] == 1]
                predictive_result = [i for i in actual_num if out1b[i] == 1]
                recall = len(predictive_result) / len(actual_num)
                predictive_true_count = [i for i in predictive1_count if i in actual_num]
                pre = len(predictive_true_count) / len(predictive1_count)
                actual_num0 = [i for i in range(len(wc)) if wc[i] == 0]
                predictive_result0 = [i for i in actual_num0 if out1b[i] == 0]
                sensitive = len(predictive_result0) / len(actual_num0)

                pre_1_index = [p_yb[i] for i in range(len(wc)) if out1b[i] == 1]
                pre_0_index = [p_yb[i] for i in range(len(wc)) if out1b[i] == 0]
                pre_1_actual_0 = [p_yb[i] for i in range(len(wc)) if out1b[i] == 1 and wc[i] == 0]
                pre_0_actual_1 = [p_yb[i] for i in range(len(wc)) if out1b[i] == 0 and wc[i] == 1]
                #print(pre_1_index)
                #print(pre_0_index)
                #print(pre_1_actual_0)
                #print(pre_0_actual_1)

                #pre_1_actual_0_index = [train_index[i] for i in range(len(wc1)) if out1[i] == 1 and wc1[i] == 0 and p_y[i]>0.97]

                print(
                    f"fold{i}  auroc is {best_roc:.4f} aupr is {best_f1:.4f} accuracy is {best_acc:.4f} recall is {recall:.4f},precision is {pre:.4f},sensitivity is {sensitive:.4f} ")
                for i in range(101):
                    model = Predictor(hidden_size, dropout, device
                                      , dti_cl).to(args['device'])
                    # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
                    optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())

                    wc1 = labelcnm.reshape(-1)
                    pre_1_actual_0_index = [train_index[i] for i in range(len(wc1)) if p_y[i] >=cnm]#0.99555555
                    for i in pre_1_actual_0_index:
                        label[i] = 1
                    for epoch in range(epochs1):
                        # fold1  auroc is 0.9316 aupr is 0.9256 accuracy is 0.8742 recall is 0.8626,precision is 0.8555,sensitivity is 0.8545 ,zheng,0.9316 ,0.9256,0.8742
                        # loss, train_acc, task1_roc, acc, task1_roc1, task1_pr,out1,labelcnm,p_y,p_yy,zzjj,dasbb = train(model, optim, data1,feature, edge,train_index, test_index, label)
                        # loss, train_acc, task1_roc, acc, task1_roc1, task1_pr ,recall, pre, sensitive= train(model, optim, data1, feature, edge,train_index, test_index, label)
                        loss, train_acc, task1_roc, acc1, task1_roc11, task1_pr1, out1, labelcnm, p_y, p_yy, zzjj, dasbb, acc, task1_roc1, task1_pr, labelcnmb, out1b, p_yb = train(
                            model, optim, data1, feature, edge, train_index, train_index, test_index, label, is_ce)
                        if acc > best_acc1:
                            best_acc1 = acc
                        if task1_pr > best_f11:
                            best_f11 = task1_pr
                        if task1_roc1 > best_roc1:
                            best_roc1 = task1_roc1

                            # torch.save(obj=model.state_dict(), f=f"{dir}/net.pth")
                    all_acc1.append(best_acc1)
                    all_roc1.append(best_roc1)
                    all_f11.append(best_f11)
                    # all_pre.append(best_pre)
                    # all_sens.append(best_sens)
                    # all_recall.append(best_recall)
                    # print(f"fold{i}  auroc is {best_roc:.4f} aupr is {best_f1:.4f} accuracy is {best_acc:.4f} recall is {best_recall:4f} precision is {best_pre:4f} sensitivity is {best_sens:4f}")
                    predictive1_count = [i for i in range(len(out1b)) if out1b[i] == 1]
                    wc = labelcnmb.reshape(-1)
                    actual_num = [i for i in range(len(wc)) if wc[i] == 1]
                    predictive_result = [i for i in actual_num if out1b[i] == 1]
                    recall1 = len(predictive_result) / len(actual_num)
                    predictive_true_count = [i for i in predictive1_count if i in actual_num]
                    pre1 = len(predictive_true_count) / len(predictive1_count)
                    actual_num0 = [i for i in range(len(wc)) if wc[i] == 0]
                    predictive_result0 = [i for i in actual_num0 if out1b[i] == 0]
                    sensitive1 = len(predictive_result0) / len(actual_num0)

                    pre_1_index = [p_yb[i] for i in range(len(wc)) if out1b[i] == 1]
                    pre_0_index = [p_yb[i] for i in range(len(wc)) if out1b[i] == 0]
                    pre_1_actual_0 = [p_yb[i] for i in range(len(wc)) if out1b[i] == 1 and wc[i] == 0]
                    pre_0_actual_1 = [p_yb[i] for i in range(len(wc)) if out1b[i] == 0 and wc[i] == 1]
                    # print(pre_1_index)
                    # print(pre_0_index)
                    # print(pre_1_actual_0)
                    # print(pre_0_actual_1)
                    wc1 = labelcnm.reshape(-1)

                    print(
                        f"fold{i}  auroc is {best_roc1:.4f} aupr is {best_f11:.4f} accuracy is {best_acc1:.4f} recall is {recall1:.4f},precision is {pre1:.4f},sensitivity is {sensitive1:.4f} ")




            # print(
            # f"{name},{sum(all_acc) / len(all_acc):.4f},  {sum(all_roc) / len(all_roc):.4f} ,{sum(all_f1) / len(all_f1):.4f}")
            print(
                f"{name},{sum(all_roc) / len(all_roc):.4f} ,{sum(all_f1) / len(all_f1):.4f},{sum(all_acc) / len(all_acc):.4f}")

            print(
                f"{name},{sum(all_roc1) / len(all_roc1):.4f} ,{sum(all_f11) / len(all_f11):.4f},{sum(all_acc1) / len(all_acc1):.4f}")


        # 20:zheng,,0.8586,  0.9250 ,0.9216（没有引入异构信息）    0.8385,  0.9282 ,0.9235（完整）   zheng,0.8586,  0.9279 ,0.9240（加权重1：1：0.2）
        def train(model, optim, data, feature, edge, train_index, test_index, test_index1, correct_interaction,tjkst):
            model.train()
            # out, cl_loss, d, p = model(data,feature, edge,train_index,test_index,correct_interaction)
            loss, train_true, predicted_interaction, correct_labels, predicted_labels, predicted_scores, dasb, correct_labels111, predicted_labels111, predicted_scores111 = model(
                data, feature, edge, train_index, test_index, test_index1, correct_interaction,tjkst)
            # cl_loss = cl_loss.to(torch.long)
            reg = get_L2reg(model.parameters())
            # loss=loss+1e-4*reg
            # loss=loss+1**reg
            loss = loss + 1e-5 ** reg
            # loss = loss +reg
            #print(loss)
            # out=out.to(torch.long)
            train_acc = (predicted_interaction.argmax(dim=1) == train_true.reshape(-1)).sum(dtype=float) / len(
                train_index)

            task1_roc = get_roc(predicted_interaction, train_true)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print(f"{epoch} epoch loss  {loss:.4f} train is acc  {train_acc:.4f}, task1 roc is {task1_roc:.4f},")
            te_acc1, te_task1_roc11, te_task1_pr1, te_acc, te_task1_roc1, te_task1_pr = main_test(predicted_scores,
                                                                                                  correct_labels,
                                                                                                  test_index,
                                                                                                  predicted_labels,
                                                                                                  predicted_scores111,
                                                                                                  correct_labels111,
                                                                                                  test_index1,
                                                                                                  predicted_labels111)
            # te_acc, te_task1_roc1, te_task1_pr,recall, pre, sensitive=main_test1(predicted_scores, correct_labels, test_index, predicted_labels)
            return loss.item(), train_acc, task1_roc, te_acc1, te_task1_roc11, te_task1_pr1, predicted_labels, correct_labels, predicted_scores, predicted_interaction, train_true, dasb, te_acc, te_task1_roc1, te_task1_pr, correct_labels111, predicted_labels111, predicted_scores111
            # return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr,recall, pre, sensitive


        def main_test(out, label, test_index, out1, out1nmb, label1, test_index1, out11):
            # print('out',out)
            acc1 = (out1 == label.reshape(-1)).sum(dtype=float) / len(test_index)
            task_roc = get_roc1(out, label)

            task_pr = get_pr1(out, label)

            acc11 = (out11 == label1.reshape(-1)).sum(dtype=float) / len(test_index1)
            task_roc1 = get_roc1(out1nmb, label1)

            task_pr1 = get_pr1(out1nmb, label1)
            return acc1, task_roc, task_pr, acc11, task_roc1, task_pr1


        def main_test1(out, label, test_index, out1):
            # print('out',out)
            acc1 = (out1 == label.reshape(-1)).sum(dtype=float) / len(test_index)
            wc = label.reshape(-1)
            print('out1', out1)
            predictive1_count = [i for i in range(len(out1)) if out1[i] == 1]
            actual_num = [i for i in range(len(wc)) if wc[i] == 1]
            predictive_result = [i for i in actual_num if out1[i] == 1]
            recall = len(predictive_result) / len(actual_num)
            predictive_true_count = [i for i in predictive1_count if i in actual_num]
            pre = len(predictive_true_count) / len(predictive1_count)

            actual_num0 = [i for i in range(len(wc)) if wc[i] == 0]
            predictive_result0 = [i for i in actual_num0 if out1[i] == 0]
            sensitive = len(predictive_result0) / len(actual_num0)
            task_roc = get_roc1(out, label)

            task_pr = get_pr1(out, label)
            return acc1, task_roc, task_pr, recall, pre, sensitive


        train_indeces, test_indeces = get_cross(dtidata, 2)
        main(train_indeces, test_indeces, seed)



