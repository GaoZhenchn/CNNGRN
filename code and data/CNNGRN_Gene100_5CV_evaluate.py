# coding: utf-8
"""
2022/01/06
anthor: Gao Zhen

"""


import warnings
warnings.filterwarnings('ignore')
# import re
import time
import os
import numpy as np
# import pandas as pd
import random
# import shutil
# from math import floor
# import matplotlib.pyplot as plt
# from keras import models,layers,regularizers
# from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve,auc,roc_auc_score
# from scipy import interp
from CNNGRN.code import CNNGRN_definedFunc


negNum = [176, 249, 195, 211, 193]  # the number of negative samples
network = ['Gene100_NET1', 'Gene100_NET2', 'Gene100_NET3', 'Gene100_NET4', 'Gene100_NET5']
network_dict = {"AUROC average: ":0,
                "AUROC std: ":0,
                "AUPR average: ": 0,
                "AUPR std: ": 0,
                "F1 average: ": 0,
                "F1 std: ": 0,
                "Acc average: ": 0,
                "Acc std: ": 0}

for net in range(5):  # five networks
    print(network[net] + 'is training...................................................................')
    # 1. read raw data
    path = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\DREAM4_GoldStandard_InSilico_Size100_' + str(net + 1) + '.tsv'
    pathts = 'D:\jupyter_project\CNNGRN\DATA\DREAM100\insilico_size100_' + str(net + 1) + '_timeseries.tsv'
    gene10, gene10_ts = CNNGRN_definedFunc.readRawData_gene100(path, pathts)

    # 2. get directed adjacency matrix
    geneNetwork = CNNGRN_definedFunc.createGRN_gene100(gene10)

    # 3. construct positive samples and negative samples
    positive_data, negative_data = CNNGRN_definedFunc.createSamples_gene100(gene10_ts, geneNetwork)

    # 4. 10 times 5CV
    kf = KFold(n_splits=5, shuffle=True)
    netavgAUROCs = []  # Store the average AUC of a network of 10 times 5CV
    netavgAUPRs = []
    netavgF1s = []
    netavgAccs = []

    for ki in range(1):
        print('\n')
        print(network[net])
        print("\nthe {} 5CV..........\n".format(ki + 1))
        # Shuffle positive and negative samples
        random.shuffle(positive_data)
        random.shuffle(negative_data)
        # The negative samples with the same number of positive samples were randomly selected and
        # combined to obtain alldata of all the training samples, and then scrambled
        num = negNum[net]
        print('the number of negative samples:' + str(num))
        alldata = np.vstack((positive_data, negative_data[0:num]))
        random.shuffle(alldata)
        # All samples were converted to obtain the sample,label,coordinates
        dataX, labelY, position = CNNGRN_definedFunc.transform_data(alldata)

        # 5CV
        AUROCs = []
        AUPRs = []
        F1s = []
        Accs = []
        for train_index, test_index in kf.split(dataX, labelY):
            # 6.1 4:1 train set:test set
            # print('train_index:%s , test_index: %s ' %(train_index,test_index))
            trainX, testX = dataX[train_index], dataX[test_index]  # testX.shape (71, 1, 620, 1)
            trainY, testY = labelY[train_index], labelY[test_index]
            positionX, positionY = position[train_index], position[test_index]
            # Remove label information from the test set
            for i in range(len(test_index)):
                row = positionY[i][0]
                col = positionY[i][1]
                testX[i][0][210 + col] = 0
                testX[i][0][520 + row] = 0

            # 6.2 bulid model
            model = CNNGRN_definedFunc.create_model_gene100()


            logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
            log_file_path = 'log\\log_' + logTime + '.csv'
            trained_models_path = 'trained_models\\'
            if (os.path.exists(trained_models_path) != 1):
                os.mkdir(trained_models_path)

            # model callbacks
            patience = 10
            early_stop = EarlyStopping('val_acc', 0.0001, patience=patience)
            reduce_lr = ReduceLROnPlateau('val_acc', factor=0.001, patience=int(patience / 2),
                                          verbose=1)
            csv_logger = CSVLogger(log_file_path, append=True)
            model_names = trained_models_path + logTime + '.{epoch:02d}-{acc:2f}.h5'
            model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', mode="max", verbose=1,
                                               save_best_only=True, save_weights_only=True)

            # callbacks = [model_checkpoint,csv_logger,early_stop,reduce_lr]
            callbacks = [csv_logger, early_stop, reduce_lr]


            (trainXX, testXX, trainYY, testYY) = train_test_split(trainX, trainY, test_size=0.2, random_state=1,
                                                                  shuffle=True)

            # train model
            history = model.fit(trainXX, trainYY, validation_data=(testXX, testYY), batch_size=4, epochs=200,
                                callbacks=callbacks)

            # 7. predict testX
            score_1 = model.predict(testX)
            # 8. calculate AUROC
            Recall, SPE, Precision, F1, MCC, Acc, aucROC, AUPR = CNNGRN_definedFunc.scores(testY[:, 1], score_1[:, 1], th=0.5)



            # ONE network
            AUROCs.append(aucROC)
            AUPRs.append(AUPR)
            F1s.append(F1)
            Accs.append(Acc)

        # one 5CV
        avg_AUROC = np.mean(AUROCs)
        avg_AUPR = np.mean(AUPRs)
        avg_F1s = np.mean(F1s)
        avg_Accs = np.mean(Accs)

        # ten 5CV
        netavgAUROCs.append(avg_AUROC)  # length = 10
        netavgAUPRs.append(avg_AUPR)
        netavgF1s.append(avg_F1s)
        netavgAccs.append(avg_Accs)

    print(network[net] + '---------------------------------------------------------------------')
    #  ten 5CV
    AUROC_mean = np.mean(netavgAUROCs) #  length = 1
    AUROC_std = np.std(netavgAUROCs, ddof=1)
    AUPR_mean = np.mean(netavgAUPRs)
    AUPR_std = np.std(netavgAUPRs)
    F1_mean = np.mean(netavgF1s)
    F1_std = np.std(netavgF1s)
    Acc_mean = np.mean(netavgAccs)
    Acc_std = np.std(netavgAccs)

    AUROC_mean = float('{:.4f}'.format(AUROC_mean))
    AUROC_std = float('{:.4f}'.format(AUROC_std))
    AUPR_mean = float('{:.4f}'.format(AUPR_mean))
    AUPR_std = float('{:.4f}'.format(AUPR_std))
    F1_mean = float('{:.4f}'.format(F1_mean))
    F1_std = float('{:.4f}'.format(F1_std))
    Acc_mean = float('{:.4f}'.format(Acc_mean))
    Acc_std = float('{:.4f}'.format(Acc_std))

    # The mean and standard deviation of AUC are saved in the dictionary network_dict
    network_dict["AUROC average: "] = AUROC_mean
    network_dict["AUROC std: "] = AUROC_std
    network_dict["AUPR average: "] = AUPR_mean
    network_dict["AUPR std: "] = AUPR_std
    network_dict["F1 average: "] = F1_mean
    network_dict["F1 std: "] = F1_std
    network_dict["Acc average: "] = Acc_mean
    network_dict["Acc std: "] = Acc_std


    network_dict_name = network[net]
    filename = open('D:\\pycharmProjects\\CNNGRN\\results\\0120GENE100EXP\\' + network_dict_name + '.txt', 'w')
    for k, v in network_dict.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()







