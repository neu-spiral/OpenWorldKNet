from __future__ import print_function

import argparse
import pickle
import numpy as np
from sklearn import cluster, metrics

import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import keras
from keras.datasets import mnist
from time import time
from scipy.stats import mode
from scipy.io import loadmat
import pickle
from tqdm import tqdm




NUM_CLASSES = 50
new_classes = [8, 9, 0, 3, 11, 4, 7, 1, 14, 6, 10, 13, 5, 2, 12]
old_classes = list(x for x in range(NUM_CLASSES) if x not in new_classes)
slice_size = 198
stride = 16
print(new_classes)
print(old_classes)


def cluster_accuracy(y_true, y_predict):
    # we assume the cluster labels are determined by its majority class inside
    # print(y_true.shape)
    # print(y_predict.shape)
    pseudo_classes = np.unique(y_predict)
    y_relabel = np.zeros_like(y_predict)
    for p_class in pseudo_classes:
        true_corresponding_classes = y_true[y_predict==p_class]
        majority = mode(true_corresponding_classes)[0]
        y_relabel[y_predict==p_class] = majority
    acc = np.sum(y_relabel == y_true) / float(len(y_true))
    return acc





def read_file(file):
    # Real hard work here
    pickle_data = pickle.load(open(file, 'rb'))
    key_len = len(pickle_data.keys())
    if key_len == 1:
        complex_data = pickle_data[pickle_data.keys()[0]]
    elif key_len == 0:
        return None, 0
    else:
        # TODO: add support to 'result' folder
        raise Exception("{} {} Key length not equal to 1!".format(file, str(pickle_data.keys())))
        pass

    if complex_data.shape[0] == 0:
        # print complex_data.shape
        return None, 0

    real_data = np.expand_dims(complex_data.real, axis=1)
    imag_data = np.expand_dims(complex_data.imag, axis=1)
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    return ex_data, samples_in_example



def normalize(device_examples,stats):
    mean_val = stats['mean']
    std_val = stats['std']
    #number_of_slices = 0 

    normalized = {}
    for device in tqdm(device_examples.keys()):
        all_normalized_example_of_device = []
        for examples in device_examples[device]:
            exp, number = read_file(examples)
            if exp is not None:
                #print(exp.shape,number)
                normalized_example = (exp - mean_val) / std_val
                #print('normalized_example.shape',normalized_example.shape)
                #number_of_slices+=(2*normalized_example.shape[0]-slice_size)/stride
                all_normalized_example_of_device.append(normalized_example)
    
        normalized[device] = all_normalized_example_of_device
    return normalized
        


def slicing(examples_normalized,ids):
    X = []
    Y = []

    for device in tqdm(examples_normalized.keys()):
        for normalized_e in examples_normalized[device]:
            normalized_e = normalized_e.reshape(normalized_e.shape[0]*normalized_e.shape[1],)
            for i in range((normalized_e.shape[0]-slice_size)/stride):
                X.append(normalized_e[i*stride:i*stride+slice_size])
                Y.append(ids[device])

    return X,Y



start = time()
with open('/scratch/RFMLS/dec18_darpa/v3_list/equalized/1Cv2/phy_payload_no_offsets_iq/device_exps.pkl') as handle:
    device_examples = pickle.load(handle)
    
with open('/scratch/RFMLS/dec18_darpa/v3_list/equalized/1Cv2/phy_payload_no_offsets_iq/device_ids.pkl') as handle:
    device_ids = pickle.load(handle)
    
with open('/scratch/RFMLS/dec18_darpa/v3_list/equalized/1Cv2/phy_payload_no_offsets_iq/stats.pkl') as handle:
    stats = pickle.load(handle)
    

print(device_ids)




devices_ids_new = {dev:id for (dev,id) in device_ids.items() if id in new_classes}
devices_ids_old = {dev:id for (dev,id) in device_ids.items() if id in old_classes}
print(devices_ids_new)
examples_old = {}
examples_new = {}
for ent in device_examples.keys():
    if ent in devices_ids_old:
        examples_old[ent] = device_examples[ent]
    elif ent in devices_ids_new:
        examples_new[ent] = device_examples[ent]
    else:
        print("Not valid")

print(len(examples_new))
print(len(examples_old))



normalized_old= normalize(examples_old,stats)
normalized_new= normalize(examples_new,stats)

print(len(normalized_old.keys()))
print(len(normalized_new.keys()))

X_old,Y_old = slicing(normalized_old,device_ids)
X_new,Y_new = slicing(normalized_new,device_ids)



X_old = np.asarray(X_old)
Y_old = np.asarray(Y_old)


X_new = np.asarray(X_new)
Y_new = np.asarray(Y_new)

print(X_old.shape,Y_old.shape)
print(X_new.shape,Y_new.shape)


print('********PCA Part***********')
number_of_feature_framewrok = 128+32

pca = PCA(n_components=number_of_feature_framewrok)
pca.fit(X_old)
Culumative_EVR = np.cumsum(pca.explained_variance_ratio_)
print("saved variance",Culumative_EVR[132])


X_old_pca = pca.transform(X_old)
X_new_pca = pca.transform(X_new)


kmeans = cluster.KMeans(n_clusters=15, random_state=0).fit(X_new_pca)
label_cluster = kmeans.labels_
y_true = Y_new
y_cluster = label_cluster

acc = cluster_accuracy(y_true, y_cluster)
ARI = metrics.adjusted_rand_score(y_true, y_cluster)
NMI = metrics.adjusted_mutual_info_score(y_true, y_cluster)
HS = metrics.homogeneity_score(y_true, y_cluster)
VM = metrics.v_measure_score(y_true, y_cluster)

print(acc,ARI,NMI,HS,VM)
end = time()
print("Executaion time",end-start)

