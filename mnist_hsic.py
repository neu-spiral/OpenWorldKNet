
from __future__ import print_function

import argparse
import pickle
import numpy as np
from sklearn import cluster, metrics

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.layers.merge import Concatenate
from keras import backend as K
import tensorflow as tf
import os

from KernelLayer import KernelLayer
from scipy.stats import mode

NUM_CLASSES, IMG_ROWS, IMG_COLS = 10, 28, 28

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


def get_data_by_label(data, target, labels=range(NUM_CLASSES), round=True, batch_size=128):
    """
    Extract data according to a list of labels
    data = x_train
    target = y_train (not one_hot)
    labels = [0, 1]
    """
    if type(labels) is int:
        labels = [labels]
    position = []
    for label in labels:
        position_part = np.where(target == label)[0] # That is because np.where returns a tuple
        position.extend(position_part)
    data_part = data[position]
    target_part = target[position]
    if round:
        length = len(position)
        length = int( int(length / batch_size) * batch_size )
        data_part = data_part[0:length]
        target_part = target_part[0:length]
    return data_part, target_part

def labels_to_categorical(labels, old_classes):
    """
    Convert nominal labels to categorical w.r.t. a list of classes
    """
    categorical_labels = keras.utils.to_categorical(labels, NUM_CLASSES)
    categorical_labels = np.delete(categorical_labels,
                                   [i for i in range(NUM_CLASSES) if i not in old_classes],
                                   axis=1)
    return categorical_labels

#def train_val_split(data, target, ratio=0.8):
#    """
#    Splits dataset into training and validation subsets in specified proportions. Class ratio is preserved
#    """
#    if 1<ratio<0 or ratio is None:
#        ratio = 0.8
#    x_train = np.zeros((data.shape))[:0]
#    x_val = np.zeros((data.shape))[:0]
#    y_train = np.zeros((target.shape))[:0]
#    y_val = np.zeros((target.shape))[:0]
#
#    labels = np.unique(target)
#    for label in labels:
#        idx = np.nonzero(target==label)
#        randperm = np.random.permutation(len(idx))
#        th = int(len(y_train)*ratio)
#        x_train = np.concatenate((x_train, data[idx[randperm[:th]]]))
#        x_val = np.concatenate((x_val, data[idx[randperm[th:]]]))
#        y_train = np.concatenate((y_train, target[idx[randperm[:th]]]))
#        y_val = np.concatenate((y_val, target[idx[randperm[th:]]]))
#
#    randperm = np.random.permutation(len(y_train))
#    x_train = x_train[randperm]
#    y_train = y_train[randperm]
#    randperm = np.random.permutation(len(y_val))
#    x_val = x_val[randperm]
#    y_val = y_val[randperm]
#    return (x_train, y_train), (x_val, y_val)

def get_models(input_shape, num_classes, lam=0.5, sigma=1, batch_size=128, num_features=128):
    """
    Create original classification model
    """
    x_input = Input(shape=input_shape, name='Input')
    x = Conv2D(32, (3,3), activation='relu')(x_input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    representation = Dense(num_features, activation='relu', name='FeaturesOld')(x)
    x_kernel = KernelLayer(batch_size=batch_size, sigma=sigma)(representation)
    #print(x_kernel)
    logits = Dense(num_classes, name='LogitsOld')(representation)
    output = Activation('softmax')(logits)

    model = Model(inputs=x_input, outputs=output)
    feature_extractor = Model(inputs=x_input, outputs=representation)
    def hsic_loss(lam):
        def loss(y_true, y_pred):
            # cross entropy
            xentropy = keras.losses.categorical_crossentropy(y_true, y_pred)
            # calculating kernel
            # kernel, n_kernel = get_guassian_kernel(representation, sigma, batch_size=batch_size)
            # print(x_kernel)
            # y_kernel = tf.einsum('ij,kj->ik', y_true, y_true)
            y_kernel = tf.tensordot(y_true, tf.transpose(y_true), 1)
            H = tf.eye(batch_size) - tf.ones((batch_size,batch_size))/float(batch_size)
            x_kernel_H = tf.matmul(x_kernel, H)
            y_kernel_H = tf.matmul(y_kernel, H)
            hsic_loss = -tf.linalg.trace(tf.matmul(x_kernel_H, y_kernel_H))
            return xentropy + lam*hsic_loss
        return loss

    model.compile(loss=hsic_loss(lam=lam),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model, feature_extractor

def get_expand_models(old_model, num_new_classes, num_new_features=32):
    """
    Expand original classification model to accomodate novel classes
    """
    x_input = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(x_input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten(name='Diverge')(x)
    # old class branch
    representation_old = Dense(old_model.get_layer('FeaturesOld').output_shape[1], activation='relu', name='FeaturesOld')(x)
    logits_old = Dense(old_model.get_layer('LogitsOld').output_shape[1], name='LogitsOld')(representation_old)
    # new class branch
    representation_new = Dense(num_new_features, activation='relu', name='FeaturesNew')(x)
    logits_new = Dense(num_new_classes, name='LogitsNew')(representation_new)
    # merge old and new class branches
    logits_all = Concatenate(axis=-1, name='LogitsAll')([logits_old, logits_new])
    output = Activation('softmax')(logits_all)

    model = Model(inputs=x_input, outputs=output)
    feature_extractor = Model(inputs=x_input, outputs=representation_new)
    branch_old = Model(inputs=x_input, outputs=logits_old)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    branch_old.set_weights(Model(inputs=old_model.input, outputs=old_model.get_layer('LogitsOld').output).get_weights())
    return model, feature_extractor

# def get_guassian_kernel(x, sigma, batch_size):
#     bs = batch_size
#     # kernel = tf.get_variable(name='kernel', shape=(bs, bs)) 
#     kernel_list = []
#     for i in range(bs):
#         dif = x[i, :] - x
#         kernel_list.append( K.exp(-K.sum(dif*dif, axis=1) / (2*sigma*sigma)) )
#     kernel = tf.stack(kernel_list, axis=1)
#     # normalization
#     ds = 1.0/K.sqrt(K.sum(kernel, axis=0))
#     D = tf.einsum('i,j->ij', ds, ds) # outer product
#     n_kernel = kernel * D
#     return kernel, n_kernel

def show_and_save_metrics(exp_name, x, x_features, y_true, y_cluster, results):
    acc_train = cluster_accuracy(y_true, y_cluster)
    ARI_train = metrics.adjusted_rand_score(y_true, y_cluster)
    AMI_train = metrics.adjusted_mutual_info_score(y_true, y_cluster, average_method='arithmetic')
    HCV_train = metrics.homogeneity_completeness_v_measure(y_true, y_cluster)
    Sil_train = metrics.silhouette_score(x_features, y_cluster, metric='euclidean', sample_size=max(min(args.sample_size,len(x)), int(len(x)/10)) if args.sample_size is not None else None)
    # ContMat_train = metrics.cluster.contingency_matrix(y_true, y_cluster)
    print('\n' + exp_name + ':\n{0:>40s} {1:.4f}\n{2:>40s} {3:.4f}\n{4:>40s} {5:.4f}\n{6:>40s} [{7:.4f}, {8:.4f}, {9:.4f}]\n{10:>40s} {11:.4f}'.format("Adjusted Rand Index:", ARI_train, "Adjusted Mutual Information:", AMI_train, "Silhouette Coefficient:", Sil_train,"Homogeneity, Completeness, V-measure:", *HCV_train, "Cluster Accuracy:", acc_train))
    results[res_entry]['ARI'].append(ARI_train)
    results[res_entry]['AMI'].append(AMI_train)
    results[res_entry]['Silhouette'].append(Sil_train)
    results[res_entry]['Homogeneity'].append(HCV_train[0])
    results[res_entry]['Completeness'].append(HCV_train[1])
    results[res_entry]['V-measure'].append(HCV_train[2])
    results[res_entry]['Cluster Acc'].append(acc_train)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Detect New Classes", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-nc', '--new_classes', type=str, default='25',
                        help='List of MNIST digits to be considered as new classes. All other classes are considered as old')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--id_gpu', type=str, default=0,
                        help='Specify which gpu to use')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size for silhouette')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Weight of the hsic loss, wrt xentropy')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Variance of rbf kernel')
    parser.add_argument('--save_path', type=str, default='/home/$USER/NewClassDetection/',
                        help='Path to directory with results')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

    new_classes = list(set(int(s) for s in args.new_classes)) # [0,6]
    new_classes.sort()
    old_classes = list(x for x in range(NUM_CLASSES) if x not in new_classes)

    filename = 'MNIST_hsic.pkl'
    try:
        results = pickle.load(open(args.save_path+filename, 'rb'))
    except EOFError:
        results = {}
    res_entry = ''.join(map(str, new_classes)) + '-' + str(args.epochs) + '-' + str(args.lam) + '-' + str(args.sigma)
    results[res_entry] = {'Accuracy':[],
                          'ARI':[],
                          'AMI':[],
                          'Silhouette':[],
                          'Homogeneity':[],
                          'Completeness':[],
                          'V-measure':[],
                          'Cluster Acc':[]}
    """
    #TODO: Check if file contains results for given new classes and # epochs
    """

    # get data and split between train, validation and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_new, y_train_new = get_data_by_label(x_train, y_train, labels=new_classes, round=True, batch_size=args.batch_size)
    x_test_new, y_test_new = get_data_by_label(x_test, y_test, labels=new_classes, round=True, batch_size=args.batch_size)
    x_train, y_train = get_data_by_label(x_train, y_train, labels=old_classes, round=True, batch_size=args.batch_size)
    x_test, y_test = get_data_by_label(x_test, y_test, labels=old_classes, round=True, batch_size=args.batch_size)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
        x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
        x_train_new = x_train_new.reshape(x_train_new.shape[0], 1, IMG_ROWS, IMG_COLS)
        x_test_new = x_test_new.reshape(x_test_new.shape[0], 1, IMG_ROWS, IMG_COLS)
        input_shape = (1, IMG_ROWS, IMG_COLS)
    else:
        x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
        x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
        x_train_new = x_train_new.reshape(x_train_new.shape[0], IMG_ROWS, IMG_COLS, 1)
        x_test_new = x_test_new.reshape(x_test_new.shape[0], IMG_ROWS, IMG_COLS, 1)
        input_shape = (IMG_ROWS, IMG_COLS, 1)

    randperm = np.random.permutation(len(x_train))
    bs = args.batch_size
    cutoff = (int(len(y_train)*.8) // bs) * bs
    y_val = y_train[randperm[cutoff:]]
    y_train = y_train[randperm[:cutoff]]
    x_val = x_train[randperm[cutoff:]].astype('float32') / 255
    x_train = x_train[randperm[:cutoff]].astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print('\nOld classes: ', old_classes)
    print('\t{0:7d} train samples\n\t{1:7d} validation samples\n\t{2:7d} test samples'.format(len(x_train), len(x_val), len(x_test)))
    x_train_new = x_train_new.astype('float32') / 255
    x_test_new = x_test_new.astype('float32') / 255
    print('\nNew classes: ', new_classes)
    print('\t{0:7d} train samples\n\t{1:7d} validation samples\n\t{2:7d} test samples'.format(int(len(x_train_new)*.8), len(x_train_new)-int(len(x_train_new)*.8), len(x_test_new)))
    print('\n\n'+'*'*75+'\n')

    # get models
    model, feature_extractor = get_models(input_shape, len(old_classes), lam=args.lam, sigma=args.sigma, batch_size=args.batch_size)

    # train classifier
    model.fit(x_train,
              labels_to_categorical(y_train, old_classes),
              batch_size=args.batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_val,
                               labels_to_categorical(y_val, old_classes)))
    score = model.evaluate(x_test,
                           labels_to_categorical(y_test, old_classes), batch_size=args.batch_size, verbose=0)
    print('\n'+'*'*75+'\n\n\nTest loss: {0:.4f}\nTest accuracy: {1:.4f}'.format(score[0], score[1]))
    results[res_entry]['Accuracy'].append(score[1])

    # cluster training samples (old classes)
    x_train_features = feature_extractor.predict(np.concatenate((x_train,x_val)))
    kmeans_train = cluster.KMeans(n_clusters=len(old_classes)).fit(x_train_features)
    y_train_cluster = kmeans_train.labels_
    show_and_save_metrics(exp_name='Old train+val', x=np.concatenate((x_train,x_val)), x_features=x_train_features, 
                            y_true=np.concatenate((y_train,y_val)), y_cluster=y_train_cluster, results=results)

    # cluster testing samples (old classes)
    x_test_features = feature_extractor.predict(x_test)
    kmeans_test = cluster.KMeans(n_clusters=len(old_classes)).fit(x_test_features)
    y_test_cluster = kmeans_test.labels_
    show_and_save_metrics(exp_name='Old test', x=x_test, x_features=x_test_features, 
                            y_true=y_test, y_cluster=y_test_cluster, results=results)

    # cluster training samples (new classes)
    x_train_new_features = feature_extractor.predict(x_train_new)
    kmeans_train_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_train_new_features)
    y_train_new_cluster = kmeans_train_new.labels_
    show_and_save_metrics(exp_name='New train', x=x_train_new, x_features=x_train_new_features, 
                            y_true=y_train_new, y_cluster=y_train_new_cluster, results=results)

    ContMat_train_new = metrics.cluster.contingency_matrix(y_train_new, y_train_new_cluster)
    print('\nContingency matrix:')
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in ContMat_train_new]))
    print('\n\n'+'*'*75+'\n')

    # cluster testing samples (new classes)
    x_test_new_features = feature_extractor.predict(x_test_new)
    kmeans_test_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_test_new_features)
    y_test_new_cluster = kmeans_test_new.labels_
    show_and_save_metrics(exp_name='New test', x=x_test_new, x_features=x_test_new_features, 
                            y_true=y_test_new, y_cluster=y_test_new_cluster, results=results)

    # re-assign new labels
    y_train_new_pred = np.zeros(y_train_new_cluster.shape)
    for i in range(len(new_classes)):
        y_train_new_pred[y_train_new_cluster == i] = new_classes[np.argmax(ContMat_train_new[:,i])]

    randperm = np.random.permutation(len(x_train_new))
    y_val_new = y_train_new[randperm[int(len(y_train_new)*.8):]]
    y_train_new = y_train_new[randperm[:int(len(y_train_new)*.8)]]
    y_val_new_pred = y_train_new_pred[randperm[int(len(y_train_new_pred)*.8):]]
    y_train_new_pred = y_train_new_pred[randperm[:int(len(y_train_new_pred)*.8)]]
    x_val_new = x_train_new[randperm[int(len(x_train_new)*.8):]]
    x_train_new = x_train_new[randperm[:int(len(x_train_new)*.8)]]

    # expand classification model
    print('\n\n'+'*'*75+'\n')
    print('Model Expansion\n')
    model_new, feature_extractor_new = get_expand_models(model, len(new_classes))
    model_new.fit(np.concatenate((x_train,x_train_new)),
                  keras.utils.to_categorical(np.concatenate((y_train,y_train_new_pred)), NUM_CLASSES),
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=1,
                  validation_data=(np.concatenate((x_val,x_val_new)),
                                   keras.utils.to_categorical(np.concatenate((y_val,y_val_new_pred)), NUM_CLASSES)))
    score_new = model_new.evaluate(np.concatenate((x_test,x_test_new)),
                                   keras.utils.to_categorical(np.concatenate((y_test,y_test_new)), NUM_CLASSES),
                                   verbose=0)
    print('\n'+'*'*75+'\n\n\nTest loss: {0:.4f}\nTest accuracy: {1:.4f}'.format(score_new[0], score_new[1]))
    results[res_entry]['Accuracy'].append(score_new[1])

    ConfMat_train = metrics.confusion_matrix(np.concatenate((y_train,y_val,y_train_new,y_val_new)),
                                             np.argmax(model_new.predict(np.concatenate((x_train,x_val,x_train_new,x_val_new))), axis=1))
    print('\nConfusion matrix (all train+val):')
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in ConfMat_train]))

    ConfMat_test = metrics.confusion_matrix(np.concatenate((y_test,y_test_new)),
                                                np.argmax(model_new.predict(np.concatenate((x_test,x_test_new))), axis=1))
    print('\nConfusion matrix (all test):')
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in ConfMat_test]))


    # cluster training samples (old classes)
    x_train_features_old = feature_extractor.predict(np.concatenate((x_train,x_val)))
    x_train_features_new = feature_extractor_new.predict(np.concatenate((x_train,x_val)))
    x_train_features = np.concatenate((x_train_features_old,x_train_features_new), axis=1)
    kmeans_train = cluster.KMeans(n_clusters=len(old_classes)).fit(x_train_features)
    y_train_cluster = kmeans_train.labels_
    show_and_save_metrics(exp_name='Old train+val', x=np.concatenate((x_train,x_val)), x_features=x_train_features, 
                            y_true=np.concatenate((y_train,y_val)), y_cluster=y_train_cluster, results=results)

    # cluster testing samples (old classes)
    x_test_features_old = feature_extractor.predict(x_test)
    x_test_features_new = feature_extractor_new.predict(x_test)
    x_test_features = np.concatenate((x_test_features_old,x_test_features_new), axis=1)
    kmeans_test = cluster.KMeans(n_clusters=len(old_classes)).fit(x_test_features)
    y_test_cluster = kmeans_test.labels_
    show_and_save_metrics(exp_name='Old test', x=x_test, x_features=x_test_features, 
                            y_true=y_test, y_cluster=y_test_cluster, results=results)

    # cluster training samples (new classes)
    x_train_new_features_old = feature_extractor.predict(x_train_new)
    x_train_new_features_new = feature_extractor_new.predict(x_train_new)
    x_train_new_features = np.concatenate((x_train_new_features_old,x_train_new_features_new), axis=1)
    kmeans_train_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_train_new_features)
    y_train_new_cluster = kmeans_train_new.labels_
    show_and_save_metrics(exp_name='New train', x=x_train_new, x_features=x_train_new_features, 
                            y_true=y_train_new, y_cluster=y_train_new_cluster, results=results)

    # cluster testing samples (new classes)
    x_test_new_features_old = feature_extractor.predict(x_test_new)
    x_test_new_features_new = feature_extractor_new.predict(x_test_new)
    x_test_new_features = np.concatenate((x_test_new_features_old,x_test_new_features_new), axis=1)
    kmeans_test_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_test_new_features)
    y_test_new_cluster = kmeans_test_new.labels_
    show_and_save_metrics(exp_name='New test', x=x_test_new, x_features=x_test_new_features, 
                            y_true=y_test_new, y_cluster=y_test_new_cluster, results=results)
    

    with open(args.save_path + filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
#    TODO: Get class priors by feeding x_test, y_test to trained feature_extractor
"""

