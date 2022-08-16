# coding: utf-8
"""

2021/11/21
anthor: Gao Zhen

"""


import warnings
warnings.filterwarnings('ignore')
import re
import time
import os
import numpy as np
import pandas as pd
import random
import shutil
from math import floor
import matplotlib.pyplot as plt
from keras import models,layers,regularizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer,MinMaxScaler
# from sklearn.metrics import roc_curve,auc,roc_auc_score
from scipy import interp
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve

def transform_data(train_data):
    """
    train_data：[(sample,label,（coordinates）)]
    return :
        trainX, labelY, position
    """
    feature = []
    label_ = []
    position = []
    for i in range(len(train_data)):
        feature.append(train_data[i][0])
        label_.append(train_data[i][1])
        position.append(train_data[i][2])

    feature = np.array(feature)
    label_ = np.array(label_)
#     print(label_)
    dataX = feature[:,np.newaxis,:,np.newaxis]
    print(dataX.shape)

    # Independent thermal encoding of labels  1 0 denotes 0， 0 1 denotes 1
    lb = LabelBinarizer()
    labelY_ = lb.fit(label_)
#     print(labelY_)
    labelY_2 = lb.transform(label_)
    labelY = to_categorical(labelY_2,2)
#     print(labelY)

    position = np.array(position)
    return dataX, labelY, position




def readRawData_gene100(path, pathts):
    '''
    path  orginal GRN
    pathts  time series expression data
    '''

    gene10 = pd.read_csv(path, sep='\t', header=-1)  # 176*3
    gene10 = np.array(gene10)  # (176, 3)


    gene10_ts = pd.read_csv(pathts, sep='\t')  # (210, 100)
    gene10_ts = np.array(gene10_ts)  # （210,100）
    gene10_ts = np.transpose(gene10_ts)  # transpose (100, 210)

    return gene10, gene10_ts



def createGRN_gene100(gene10):
    rowNumber = []
    colNumber = []
    for i in range(len(gene10)):
        row = gene10[i][0]
        rownum = re.findall("\d+",row)
        rownumber = int(np.array(rownum))
        rowNumber.append(rownumber)

        col = gene10[i][1]
        colnum = re.findall("\d+",col)
        colnumber = int(np.array(colnum))
        colNumber.append(colnumber)

    geneNetwork = np.zeros((100,100))
    for i in range(len(rowNumber)):
        r = rowNumber[i]-1
        c = colNumber[i]-1
        geneNetwork[r][c] = 1
#     print(np.sum(geneNetwork))
#     SAVE geneNetwork
#     data1 = pd.DataFrame(geneNetwork)
#     data1.to_csv('D:\jupyter_project\CNNGRN\DATA\DREAM100_samples\geneNetwork_100_'+str(net+1)+'.csv')
    return geneNetwork


# （expression+Y 620 620 620 620 620）
def createSamples_gene100(gene10_ts, geneNetwork):
    sample_10_pos = []
    sample_10_neg = []
    labels_pos = []
    labels_neg = []
    positive_1_position = []
    negative_0_position = []
    for i in range(100):
        for j in range(100):
            temp11 = gene10_ts[i]  # (210,)
            temp12 = geneNetwork[i]  # (100,)
            temp21 = gene10_ts[j]  # (210,)
            temp22 = geneNetwork[:, j]  # (100,)

            temp1 = np.hstack((temp11, temp12))  # (310,)  210+100
            temp2 = np.hstack((temp21, temp22))  # (310,)
            temp = np.hstack((temp1, temp2))  # (620,)
            # temp = np.hstack((temp11, temp21))  # (420,)
            # print(temp.shape)
            label = int(geneNetwork[i][j])

            if label == 1:
                sample_10_pos.append(temp)
                labels_pos.append(label)
                positive_1_position.append((i, j))

            else:
                sample_10_neg.append(temp)
                labels_neg.append(label)
                negative_0_position.append((i, j))

    # Bind feature (sample) and label together
    positive_data = list(zip(sample_10_pos, labels_pos, positive_1_position))
    negative_data = list(zip(sample_10_neg, labels_neg, negative_0_position))

    return positive_data, negative_data





