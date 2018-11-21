##     DCFA, dynamic collaborative filtering with aesthetic features
##     author @Wenhui Yu

import numpy as np
from Library import readdata
from Library import readdata_time
from Library import evaluation_F1
from Library import evaluation_NDCG
from Library import save_result
from Library import read_feature
from numpy import *
import xlwt
import time
import json

##parameter setting
dataset = 5                         # Datasets selecting 0 to 5 for 'All', '_Women', '_Men', '_CLothes', '_Shoes', '_Jewelry' respectively
eta = 0.03                          # learning rate
I = 200                             # length of latent feature
J = 100                             # length of latent feature
top_k = [5, 10, 20, 50, 100]        # number to recommend
batch_size_train = 5000             # batch size for traing
batch_size_test = 1000              # batch size for tessing
lambda_c = 0.1                      # weighting parameter for couple matrices
lambda_r = 1.5                      # regularization coefficient
vali_test = 0                       # 0 for validate set,1 for test set
feat = [3]                          # feature selecting, 0 for CNN, 1 for AES, 2 for CH, 3 for CNN+AES
feature_length = 1000               # length of feature
epoch = 200                         # number to iteration
sample_rate = 5                     # sample sample_rate negative samples for each positive item


def d(x):
    # sigmoid function for BPR, d(x) = sigmoid(-x)
    if x > 10:
        return 0
    if x < -10:
        return 1
    if x >= -10 and x <= 10:
        return 1.0 / (1.0 + exp(x))

def get_feature(dataset):
    # to load features
    feat_list = ['CNN', 'AES', 'CH', 'CNN_AES']             # feature list
    F = read_feature(feat_list[feat[0]], dataset, Q)
    for i in range(1, len(feat)):
        F = np.hstack((F, read_feature(feat_list[feat[i]], dataset, Q)))
    return F

def get_order(Order, length):
    # combining several recommendation lists into one, for tensor-based model
    order = []
    ind = 0
    while len(order) <= length:
        for line in Order:
            if not line[ind] in order:
                order.append(line[ind])
        ind += 1
    return order

def test_DCFA(U, Vu, Vt, T, M, N, F):
    # test the effectiveness
    U = mat(U)
    Vu = mat(Vu)
    Vt = mat(Vt)
    T = mat(T)
    F = mat(F)
    M = mat(M)
    N = mat(N)
    k_num = len(top_k)
    # k_num-long lists to record F1 and NDCG
    F1 = np.zeros(k_num)
    NDCG = np.zeros(k_num)
    num_item = len(Test)

    # choose batch_size_test test samples randomly
    for i in range(batch_size_test):
        j = int(math.floor(num_item * random.random()))
        # test data: [u, [i, i, i, i], [r, r, r]], where u, i, r are for user, item, time, respectively
        u = Test[j][0]
        test_item = Test[j][1]
        # score for all users
        Order = []
        for r in Test[j][2]:
            # for each r, score all items
            UV = U[u] * Vu.T + M[u] * F.T
            VT = T[r] * Vt.T + N[r] * F.T
            UV = np.array(UV.tolist()[0])
            VT = np.array(VT.tolist()[0])
            score = (UV * VT).tolist()
            # order
            b = zip(score, range(len(score)))
            b.sort(key=lambda x: x[0])
            order = [x[1] for x in b]
            order.reverse()
            Order.append(order)
        # train samples
        train_positive = train_data_aux[u][0]
        # we have len(train_data_aux[u][1]) k-length recommendation lists for each user,
        # to compare fairly with other baselines, we combine len(train_data_aux[u][1]) k-length list to one k-length lists for each user
        # we will remove at most len(train_positive) train samples from order, so we return k+len(train_positive) items
        order = get_order(Order, top_k[-1] + len(train_positive))
        # remove the train samples from the recommendations
        for item in train_positive:
            try:
                order.remove(item)
            except:
                continue
        # we also remove the train samples from test samples
        test_item = list(set(test_item) - set(train_positive))
        # test F1 and NDCG for each k
        for i in range(len(top_k)):
            F1[i] += evaluation_F1(order, top_k[i], test_item)
            NDCG[i] += evaluation_NDCG(order, top_k[i], test_item)
    # calculate the average
    F1 = (F1 / batch_size_test).tolist()
    NDCG = (NDCG / batch_size_test).tolist()
    return F1, NDCG


def train_DCFA(eta):
    # train the model
    # initialization
    U = np.array([np.array([(random.random() / math.sqrt(I)) for j in range(I)]) for i in range(P)])
    Vu = np.array([np.array([(random.random() / math.sqrt(I)) for j in range(I)]) for i in range(Q)])
    Vt = np.array([np.array([(random.random() / math.sqrt(J)) for j in range(J)]) for i in range(Q)])
    T = np.array([np.array([(random.random() / math.sqrt(J)) for j in range(J)]) for i in range(R)])

    M = np.array([np.array([(random.random() / math.sqrt(K)) for j in range(K)]) for i in range(P)])
    N = np.array([np.array([(random.random() / math.sqrt(K)) for j in range(K)]) for i in range(R)])
    e = 10**10

    # output a result without training
    print 'iteration ', 0,
    [F1, NDCG] = test_DCFA(U, Vu, Vt, T, M, N, F)
    Fmax = 0
    if F1[0] > Fmax:
        Fmax = F1[0]
    print Fmax, 'F1: ', F1, '  ', 'NDCG1: ', NDCG
    # save to the .xls file
    save_result([' '], [''] * len(top_k), [''] * len(top_k), path_excel)
    save_result('metric', ['F1'] * len(top_k), ['NDCG'] * len(top_k), path_excel)
    save_result('Top_k', top_k, top_k, path_excel)
    save_result([' '], [''] * len(top_k), [''] * len(top_k), path_excel)
    save_result('iteration ' + str(0), F1, NDCG, path_excel)

    # the number of train samples
    Re = len(train_data)
    # split the train samples with a step of batch_size_train
    bs = range(0, Re, batch_size_train)
    bs.append(Re)

    for ep in range(0, epoch):
        print 'iteration ', ep + 1,
        eta = eta * 0.99
        # iterate all train samples in one epoch
        for i in range(0, len(bs) - 1):
            if abs(U.sum()) < e:
                # initialize dU and dC to record the gradient
                dU = np.zeros((P, I))
                dVu = np.zeros((Q, I))
                dVt = np.zeros((Q, J))
                dT = np.zeros((R, J))

                dM = np.zeros((P, K))
                dN = np.zeros((R, K))
                for re in range(bs[i], bs[i + 1]):
                    # train sample: [u, i, r]
                    p = train_data[re][0]
                    qi = train_data[re][1]
                    r = train_data[re][2]

                    UV = np.dot(U[p], Vu[qi])
                    VT = np.dot(Vt[qi], T[r])
                    MDF = np.dot(M[p], F[qi])
                    NEF = np.dot(N[r], F[qi])

                    Bi = UV + MDF
                    Ci = VT + NEF
                    Ai = Bi * Ci

                    num = 0
                    # choose sample_rate negative items, and calculate the gradient
                    while num < sample_rate:
                        qj = int(random.uniform(0, Q))
                        if (not qj in train_data_aux[p][0]) and (not qj in train_time_aux[r][1]):
                            num += 1
                            UV = np.dot(U[p], Vu[qj])
                            VT = np.dot(Vt[qj], T[r])
                            MDF = np.dot(M[p], F[qj])
                            NEF = np.dot(N[r], F[qj])

                            Bj = UV + MDF
                            Cj = VT + NEF
                            Aj = Bj * Cj

                            Bij = Bi - Bj
                            Cij = Ci - Cj
                            Aij = Ai - Aj

                            dU[p] += d(Aij) * (Ci * Vu[qi] - Cj * Vu[qj]) + lambda_c * d(Bij) * (Vu[qi] - Vu[qj])
                            dVu[qi] += d(Aij) * Ci * U[p] + lambda_c * d(Bij) * U[p]
                            dVu[qj] -= d(Aij) * Cj * U[p] + lambda_c * d(Bij) * U[p]
                            dM[p] += d(Aij) * (Ci * F[qi] - Cj * F[qj]) + lambda_c * d(Bij) * (F[qi] - F[qj])
                            dVt[qi] += d(Aij) * Bi * T[r] + lambda_c * d(Cij) * T[r]
                            dVt[qj] -= d(Aij) * Bj * T[r] + lambda_c * d(Cij) * T[r]
                            dT[r] += d(Aij) * (Bi * Vt[qi] - Bj * Vt[qj]) + lambda_c * d(Cij) * (Vt[qi] - Vt[qj])
                            dN[r] += d(Aij) * (Bi * F[qi] - Bj * F[qj]) + lambda_c * d(Cij) * (F[qi] - F[qj])

                # update the matrices
                U += eta * (dU - lambda_r * U)
                Vu += eta * (dVu - lambda_r * Vu)
                Vt += eta * (dVt - lambda_r * Vt)
                T += eta * (dT - lambda_r * T)
                M += eta * (dM - lambda_r * M)
                N += eta * (dN - lambda_r * N)

        if abs(U.sum()) < e:
            [F1, NDCG] = test_DCFA(U, Vu, Vt, T, M, N, F)
            if F1[0] > Fmax:
                Fmax = F1[0]
            print Fmax, 'F1: ', F1, '  ', 'NDCG1: ', NDCG
            save_result('iteration ' + str(ep + 1), F1, NDCG, path_excel)
        else:
            break
        #return U, Vu, Vt, T, M, N
    if abs(U.sum()) < e:
        return 0
    else:
        return 1

def save_parameter():
    # record the parameters
    dataset_list = ['all', '_Women', '_Men', '_CLothes', '_Shoes', '_Jewelry']
    excel = xlwt.Workbook()
    table = excel.add_sheet('A Test Sheet')
    table.write(0, 0, 'model')
    table.write(0, 2, 'DCFA')
    table.write(1, 0, 'dataset')
    table.write(1, 2, dataset_list[dataset])
    table.write(2, 0, 'eta')
    table.write(2, 2, eta)
    table.write(3, 0, 'I')
    table.write(3, 2, I)
    table.write(4, 0, 'J')
    table.write(4, 2, J)
    table.write(5, 0, 'top_k')
    for i in range(len(top_k)):
        table.write(5, 2 + i, top_k[i])
    table.write(6, 0, 'batch_size_train')
    table.write(6, 2, batch_size_train)
    table.write(7, 0, 'batch_size_test')
    table.write(7, 2, batch_size_test)
    table.write(8, 0, 'lambda_c')
    table.write(8, 2, lambda_c)
    table.write(9, 0, 'lambda_r')
    table.write(9, 2, lambda_r)
    table.write(10, 0, 'vali_test')
    table.write(10, 2, vali_test)
    table.write(11, 0, 'feat')
    for i in range(len(feat)):
        table.write(11, 2 + i, feat[i])
    table.write(12, 0, 'fea_len')
    table.write(12, 2, feature_length)
    table.write(13, 0, 'epoch')
    table.write(13, 2, epoch)
    table.write(17, 0, ' ')

    excel.save(path_excel)

def print_parameter():
    print 'model', 'DCFA'
    print 'dataset', dataset
    print 'eta', eta
    print 'I', I
    print 'J', J
    print 'top_k', top_k
    print 'batch_size_train', batch_size_train
    print 'batch_size_test', batch_size_test
    print 'lambda_c', lambda_c
    print 'lambda_r', lambda_r
    print 'vali_test', vali_test
    print 'feat', feat
    print 'feature_length', feature_length
    print 'epoch', epoch
    print


'''*************************main function****************************'''
'''*************************main function****************************'''
for i in range(1):
    # datasets
    dataset_list = ['', '_Women', '_Men', '_CLothes', '_Shoes', '_Jewelry']
    # load data
    [train_data, train_data_aux, validate_data, test_data, P, Q] = readdata(dataset_list[dataset])
    # load data for tensor factorization
    [train_record_aux, train_time_aux, R] = readdata_time(dataset_list[dataset])
    # load features
    F = get_feature(dataset_list[dataset])
    K = len(F[0])
    # select test set or validation set
    if vali_test == 0:
        Test = validate_data
    else:
        Test = test_data

    for j in range(1):
        path_excel = 'E:\\experiment_result\\' + dataset_list[dataset] + '_DCFA_' + str(int(time.time())) + str(int(random.uniform(100,900))) + '.xls'
        save_parameter()
        print_parameter()
        train_DCFA(eta)

