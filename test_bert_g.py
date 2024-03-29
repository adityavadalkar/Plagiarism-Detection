import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import get_evaluation
from src.dataset_bert_4 import MyDataset
import argparse
import shutil
import csv
import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy as np
from src.bert_average import Bert_cls_av
from src.bert_han_g import HierAttNet
from src.bert_han_sg_g import HierGraphAttNet 
from torch.nn import CosineSimilarity
from src.MUSEAttention import MUSEAttention
from src.gmlp import gMLP
models_class = {'Bert_avg':Bert_cls_av, 'Bert_han_g':HierAttNet, 'Bert_han_sg_g':HierGraphAttNet}

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="data/cite_acl/test_cite.csv")
    parser.add_argument("--pre_trained_model", type=str, default="acl_bert_model/Bert_han_g.pth")
    parser.add_argument("--word2vec_path", type=str, default="data/word_embedding/glove.6B.50d.txt")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50) 
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--model_type", type=str, default='Bert_han_g')
    parser.add_argument("--sent_level", type=int, default=0) 
    args = parser.parse_args()
    return args

def pos_cal_mpr(pred_pos, true_pos, label_list):
    count = 0
    mpr = 0
    pred_pos = torch.cat(pred_pos, dim=1)
    _, pred_doc_pos = pred_pos.sort(dim=0,descending=True)
    pred_doc_pos = pred_doc_pos.transpose(0,1)
    pred_doc_pos = pred_doc_pos.numpy()
    pred_doc_pos = list(pred_doc_pos)
    for pred, true, label in zip(pred_doc_pos, true_pos, label_list):
        if label!=0 and true!=[]: 
            rank = []  
            for i in true:
                #print(list(pred))
                try:
                    rank.append(list(pred).index(i)+1)
                except:
                    rank.append(21)
            rank_f = min(rank)     
            mpr += 1/rank_f
            count +=1
    return mpr/count

def pos_accuracy(pred_pos, true_pos, label_list,top_num):
    true_count = 0
    count = 0
    print(label_list.shape)
    pred_pos = torch.cat(pred_pos, dim=1)
    print(pred_pos.shape)
    print(pred_pos.transpose(0,1)[6:10])
    _, pred_doc_pos = (pred_pos).topk(top_num,dim=0)
    pred_doc_pos = pred_doc_pos.transpose(0,1)
    pred_doc_pos = pred_doc_pos.numpy()
    pred_doc_pos = list(pred_doc_pos)
    print(true_pos[6:20])
    print(pred_doc_pos[6:20])
    print(label_list[6:20])
    for pred, true, label in zip(pred_doc_pos, true_pos, label_list):
        if label!=0:
            pred = set(pred)
            true = set(true)
            if pred.intersection(true) != set():
                true_count += 1
                count += 1
            else:
                count +=1
    return true_count/count

def Sort_Tuple(tup): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of 
    # sublist lambda has been used 
    return(sorted(tup, key=lambda x: float(x[1]), reverse=True))

def test(opt):
    test_params = {"batch_size": opt.batch_size,
                "shuffle": False,
                "drop_last": False}
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    if torch.cuda.is_available():
        freeze=True
        model = models_class[opt.model_type](vector_size=1024) 
        model.load_state_dict(torch.load(opt.pre_trained_model), strict=False)
    else:
        model = torch.load(opt.pre_trained_model, map_location=lambda storage, loc: storage)
    test_set = MyDataset(opt.data_path, opt.max_len)
    pos = test_set.get_pos()
    test_generator = DataLoader(test_set, **test_params)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    te_label_ls = []
    te_pred_ls = []
    te_pos_ls = []
    te_true_post_ls = []
    scores=[]
    
    mlp_graph = nn.Linear(4* 50, 2*50).cuda()
    gmlp = gMLP(len_sen=11,dim=100,d_ff=100)
    cos = CosineSimilarity()
    # total, count = 0, 0
    # attention = MUSEAttention(d_model=100, d_k=100, d_v=100, h=8)
    for te_feature1, te_feature2, te_label, _ in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature1 = te_feature1.cuda()
            te_feature2 = te_feature2.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature1, te_feature2)
            output_1 = model.encode(te_feature1)
            output_2 = model.encode(te_feature2)
            attention_x = gmlp(output_1)
            attention_y = gmlp(output_2)
            print("att x ", attention_x.shape)
            output_x = torch.cat((output_1, attention_y), dim=2)
            output_y = torch.cat((output_2, attention_x), dim=2)
            #output_x = input_1+attention_y
            #output_y = input_2+attention_x
            output_x = mlp_graph(output_x)
            output_y = mlp_graph(output_y)
            print("output x ", output_x.shape)
            
            doc_te_predictions = te_predictions[-1]
            pos_predictions = te_predictions[:-1]

            print("label: ",te_label[0])
            #te_predictions = F.softmax(te_predictions) #do not know what it is doing?
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(doc_te_predictions.clone().cpu())
        te_pos_ls.append(pos_predictions.clone().cpu())
        # break 
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    print("accuracy sim: ", ((count/total)*100))
    te_label = np.array(te_label_ls)
    te_pred = np.where(te_pred > 0.5, 1, 0)
    if opt.sent_level==1: 
        pos_acc_10 = pos_accuracy(te_pos_ls, pos, te_label, 10)
        pos_acc_5 = pos_accuracy(te_pos_ls, pos, te_label, 5)
        pos_mpr = pos_cal_mpr(te_pos_ls, pos, te_label)
    else:
        pos_acc_10 = 0
        pos_acc_5 = 0
        pos_mpr = 0
    fieldnames = ['True label', 'Predicted label', 'Content1', 'Content2']
    with open(opt.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(te_label, te_pred, test_set.texts):
            writer.writerow(
                {'True label': i, 'Predicted label':j , 'Content1': k[0], 'Content2':k[1]})

    test_metrics = get_evaluation(te_label, te_pred,
                                list_metrics=["accuracy", "loss", "confusion_matrix","f1"])
    print("Prediction:\nLoss: {} Accuracy: {} Pos Acc 10: {} Pos Acc 5:{}  mpr: {} f1: {}\nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                            test_metrics["accuracy"],
                                                                            pos_acc_10,pos_acc_5,
                                                                            pos_mpr,
                                                                            test_metrics["f1"],
                                                                            test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
