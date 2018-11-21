import numpy as np
from numpy import *
from xlrd import open_workbook
from xlutils.copy import copy
import json

def evaluation_F1(order, top_k, positive_item):
    e = 0.00000000000001
    top_k_items = set(order[0: top_k])
    positive_item = set(positive_item)
    precision = len(top_k_items & positive_item) / (len(top_k_items) + e)
    recall = len(top_k_items & positive_item) / (len(positive_item) + e)
    F1 = 2 * precision * recall / (precision + recall + e)
    return F1

def evaluation_NDCG(order, top_k, positive_item):
    top_k_item = order[0: top_k]
    e = 0.0000000001
    Z_u = 0
    temp = 0
    for i in range(0, top_k):
        Z_u += 1 / log2(i + 2)
        if top_k_item[i] in positive_item:
            temp += 1 / log2(i + 2)
    NDCG = temp / (Z_u + e)
    return NDCG

def readdata(dataset):
    #file paths
    path_train = 'E:\dataset\interactions' + dataset + '_train.json'
    path_train_aux = 'E:\dataset\interactions' + dataset + '_train_aux.json'
    path_validate = 'E:\dataset\interactions' + dataset + '_validate.json'
    path_test = 'E:\dataset\interactions' + dataset + '_test.json'
    # read files
    with open(path_train) as f:
        line = f.readline()
        train_data = json.loads(line)
    f.close()
    P = 0
    Q = 0
    for [u, i, r] in train_data:
        if u > P:
            P = u
        if i > Q:
            Q = i
    with open(path_train_aux) as f:
        line = f.readline()
        train_data_aux = json.loads(line)
    f.close()
    with open(path_validate) as f:
        line = f.readline()
        validate_data = json.loads(line)
    f.close()
    with open(path_test) as f:
        line = f.readline()
        test_data = json.loads(line)
    f.close()
    return train_data, train_data_aux, validate_data, test_data, P + 1, Q + 1

def readdata_time(dataset):
    #file paths
    path_train_record_aux = 'E:\dataset\interactions' + dataset + '_train_record_aux.json'
    path_train_time_aux = 'E:\dataset\interactions' + dataset + '_train_time_aux.json'
    # read files
    with open(path_train_record_aux) as f:
        line = f.readline()
        train_record_aux = json.loads(line)
    f.close()
    with open(path_train_time_aux) as f:
        line = f.readline()
        train_time_aux = json.loads(line)
    f.close()
    return train_record_aux, train_time_aux, len(train_time_aux)

def read_feature(feature, dataset, Q):
    path_feature = 'E:\\dataset\\features\\' + feature + '_feature.txt'
    path_dict = 'E:\dataset\id2num_dict\id2num_dict' + dataset + '.json'
    with open(path_dict) as f:
        line = f.readline()
        item_i2num_dict = json.loads(line)
    f.close()
    f = open(path_feature, 'r')
    line = eval(f.readline())
    feature = line[1]
    K = len(feature)
    F = np.zeros((Q, K))
    for i in range(0, Q):
        F[i] = feature
    for line in f:
        line = eval(line)
        item_id = line[0]
        feature = line[1]
        try:
            item_num = item_i2num_dict[item_id]
            F[item_num] = feature
        except:
            continue
    return F

def save_result(intro, F1, NDCG, path):
    rexcel = open_workbook(path)
    rows = rexcel.sheets()[0].nrows
    excel = copy(rexcel)
    table = excel.get_sheet(0)
    row = rows
    table.write(row, 0, intro)
    #table.write(row, 2, 'F1')
    for i in range(len(F1)):
        table.write(row, i + 3, F1[i])
    #table.write(row, len(F1) + 4, 'NDCG')
    for i in range(len(NDCG)):
        table.write(row, i + len(F1) + 5, NDCG[i])
    excel.save(path)