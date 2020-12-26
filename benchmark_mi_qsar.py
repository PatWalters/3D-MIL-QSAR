#!/usr/bin/env python

import sys
import os
from utils import calc_3d_pmapper, scale_descriptors
from sklearn.model_selection import train_test_split
from miqsar.estimators.wrappers import MIWrapperMLPRegressor, miWrapperMLPRegressor
from miqsar.estimators.attention_nets import AttentionNetRegressor
from miqsar.estimators.mi_nets import MINetRegressor, miNetRegressor
from sklearn.metrics import r2_score
import shutil
import numpy as np
import pickle
from glob import glob
import pandas as pd
from process_splits import generate_splits

n_epoch = 500
batch_size = 128
lr = 0.001
weight_decay = 0.01
seed = 42
init_cuda = True
ncpu = 5
random_state = 42

res = []

split_dict = generate_splits("cv_splits.csv")

# use scffold (SCAF) or random (RND) splits
split_to_use = "SCAF" 

for dataset_file in sorted(glob("pw_test/*.smi")):
    path_name, dataset_name = os.path.split(dataset_file)
    print(dataset_name)
#    split_dict = pickle.load(open("splits.pickle","rb"))
    split_list = split_dict[dataset_name.replace(".smi","")][split_to_use]

    try:
        shutil.rmtree('descriptors')
    except FileNotFoundError:
        pass
    os.mkdir('descriptors')
    path = os.path.join('descriptors', 'tmp')
    os.mkdir(path)

    bags_si, labels_si, molid_si = calc_3d_pmapper(dataset_file, nconfs=1, stereo=False, path=path, ncpu=ncpu)
    # skip this dataset if not all molecules converted to 3D
    if bags_si.shape[0] != len(split_list[0][0])+len(split_list[0][1]):
        print("Not all molecules converted to 3D")
        continue

    bags_mi, labels_mi, molid_mi = calc_3d_pmapper(dataset_file, nconfs=5, stereo=False, path=path, ncpu=ncpu)

    for train_idx, test_idx in split_list:
        #train_idx, test_idx = train_test_split(range(0,len(labels_si)))

        x_train = np.take(bags_si,train_idx,axis=0)
        x_test = np.take(bags_si,test_idx,axis=0)
        y_train  = np.take(labels_si,train_idx,axis=0)
        y_test = np.take(labels_si,test_idx,axis=0)
        molid_train = np.take(molid_si,train_idx,axis=0)
        molid_test = np.take(molid_si,test_idx,axis=0)

        #x_train, x_test, y_train, y_test, molid_train, molid_test = train_test_split(bags, labels, molid, random_state=random_state)

        x_train, x_test = scale_descriptors(x_train, x_test)

        ndim = (x_train[0].shape[-1], 256, 128, 64)

        att_net = MINetRegressor(ndim=ndim, init_cuda=init_cuda)
        att_net.fit(x_train, y_train, n_epoch=n_epoch, dropout=0.9, batch_size=batch_size, weight_decay=weight_decay, lr=lr)

        predictions = att_net.predict(x_test)

        r2 = r2_score(y_test, predictions)
        print('3D/SI/Net: r2_score = {:.2f}'.format(r2))
        res.append([dataset_name,'3D/SI/Net',r2])

        x_train = np.take(bags_mi,train_idx,axis=0)
        x_test = np.take(bags_mi,test_idx,axis=0)
        y_train  = np.take(labels_mi,train_idx,axis=0)
        y_test = np.take(labels_mi,test_idx,axis=0)
        molid_train = np.take(molid_mi,train_idx,axis=0)
        molid_test = np.take(molid_mi,test_idx,axis=0)

        #x_train, x_test, y_train, y_test, molid_train, molid_test = train_test_split(bags, labels, molid, random_state=random_state)
        x_train, x_test = scale_descriptors(x_train, x_test)

        # train 3D/MI/Bag-AttentionNet
        ndim = (x_train[0].shape[-1], 256, 128, 64)
        det_ndim = (64,)

        att_net = AttentionNetRegressor(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)
        att_net.fit(x_train, y_train, n_epoch=n_epoch, dropout=0.8, batch_size=batch_size, weight_decay=weight_decay, lr=lr)

        predictions = att_net.predict(x_test)
        instance_weights = att_net.get_instance_weights(x_test)

        r2 = r2_score(y_test, predictions)                    
        print('3D/MI/Bag-AttentionNet: r2_score = {:.2f}'.format(r2))
        res.append([dataset_name,'3D/MI/Bag-AttentionNet',r2])

res_df = pd.DataFrame(res,columns=["Dataset","Method","R2"])
res_df.to_csv("res.csv",index=False,float_format="%0.2f")


