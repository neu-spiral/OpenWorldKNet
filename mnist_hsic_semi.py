
from __future__ import print_function

import argparse
import pickle
import numpy as np
from sklearn import cluster, metrics

import keras
from keras.datasets import mnist, fashion_mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, Reshape, UpSampling2D
from keras.layers.merge import Concatenate
from keras import backend as K
import tensorflow as tf

import os
import time

from KernelLayer import KernelLayer
from scipy.stats import mode

from helpers.classifier import kmeans
from helpers.distances import median_of_pairwise_distance
from helpers.kernel_lib import getLaplacian, L_to_U, center_matrix, normalized_rbk_sklearn


NUM_CLASSES, IMG_ROWS, IMG_COLS = 10, 28, 28

# global variable
db = {
    'Ku' : None, # numpy version of the kernel of spectral embedding 
    'U' : None, # numpy version of the spectral embedding
    'Ku_batch' : np.ones((32, 32)), # numpy version of the spectral embedding of current batch
    'use_degree_matrix' : False, 

}

def update_Ku():
    Ku = db['U'].dot(db['U'].T)
    Y = center_matrix(Ku)
    if db['use_degree_matrix']: Y = db['D_inv'].dot(Y).dot(db['D_inv'].T)
    np.fill_diagonal(Y, 0)
    db['Ku'] = Y

def update_Ku_batch(indices):
    temp = db['Ku'][indices, :]
    temp = temp[:, indices]
    temp = temp.astype(np.float32)
    db['Ku_batch'] = temp

def update_U(phi_x):
    [DKxD, db['D_inv']] = normalized_rbk_sklearn(phi_x, db['sigma'])
    HDKxDH = center_matrix(DKxD)

    # TODO: if use delta kernel
    [U, db['U_normalized']] = L_to_U(NUM_CLASSES, HDKxDH)
    db['U'] = U

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

def subsample_classes(x, y, samples_per_class):
    # Here we need this method as the whole kernel may not fit in the memory
    classes = np.unique(y)
    position_dict = {x:[] for x in classes}
    classes = sorted(classes)
    max_per_class = 0
    for label in classes:
        position_part = np.where(y == label)[0]
        if len(position_part) >= max_per_class: max_per_class = len(position_part)
        position_dict[label] = position_part
    samples_per_class = samples_per_class if (samples_per_class <= max_per_class) else max_per_class
    position = []
    for label in classes:
        position.extend(position_dict[label][:samples_per_class])
    return x[position], y[position]

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

def initialize_embedding(x, y, num_clusters, sig_ratio):
    print('\tComputing initial U for Spectral Clustering...')
    start_time = time.time() 
    [allocation, km_nmi] = kmeans(num_clusters, x, y)
    raw_kmeans_time = time.time() - start_time
    print('\t\tInitial K-means NMI on raw data : %.3f, time : %.3f'%(km_nmi, raw_kmeans_time))

    # -----------------------
    start_time = time.time()
    N = x.shape[0]
    H = np.eye(N) - (1.0/N)*np.ones((N, N))
    x_mpd = float(median_of_pairwise_distance(x))
    sig = float(x_mpd * sig_ratio)
    db['sigma'] = sig

    [L, D_inv] = getLaplacian(x, sig, H=H)	
    [U, U_normalized] = L_to_U(num_clusters, L)
    
    [allocation, init_spectral_nmi] = kmeans(num_clusters, U_normalized, y)

    init_spectral_clustering_time = time.time() - start_time
    # global save
    db['U'] = U
    db['D_inv'] = D_inv
    db['U_normalized'] = U_normalized
    print('\t\tInitial Spectral Clustering NMI on raw data : %.3f, sigma: %.3f , sigma_ratio: %.3f , time : %.3f'%(init_spectral_nmi, sig, sig_ratio, init_spectral_clustering_time))

    # return U, U_normalized

def get_models(input_shape, num_classes, lam=0.5, sigma=1, batch_size=128, num_features=128):
    """
    Create original classification model
    """
    x_input = Input(shape=input_shape, name='Input') # bs, 28, 28, 1
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x_input) # bs, 28, 28, 32
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x) # bs, 28, 28, 64
    x = MaxPooling2D(pool_size=(2,2))(x) # bs, 14, 14, 64
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    representation = Dense(num_features, activation='relu', name='FeaturesOld')(x)

    # add autoencoder pretrain
    x_rev = Dense(12544, activation='relu')(representation)
    x_rev = Reshape((14, 14, 64))(x_rev)
    x_rev = UpSampling2D((2, 2))(x_rev) # 28, 28, 64
    x_rev = Conv2D(32, (3, 3), activation='relu', padding='same')(x_rev)
    x_rev = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x_rev)

    ae = Model(inputs=x_input, outputs=x_rev)
    ae.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta())

    kernel_layer = KernelLayer(batch_size=batch_size, sigma=sigma)
    x_kernel = kernel_layer(representation)
    #print(x_kernel)
    logits = Dense(num_classes, name='LogitsOld')(representation)
    output = Activation('softmax')(logits)

    model_xentropy = Model(inputs=x_input, outputs=output)
    feature_extractor = Model(inputs=x_input, outputs=representation)
    model_xentropy.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    kernel_checker = Model(inputs=x_input, outputs=x_kernel)
    # Now we need 2 more versions of the model
    # 1. supervised, only on old data
    # 2. unsupervised, on old + new data
    model_supervised = Model(inputs=x_input, outputs=output)
    model_supervised.summary()
    model_unsupervised = Model(inputs=x_input, outputs=output)
    def supervised_hsic_loss(lam):
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
            return lam*xentropy + hsic_loss
        return loss

    def unsupervised_hsic_loss(lam):
        def loss(y_true, y_pred):
            # Ku_batch = tf.convert_to_tensor(db['Ku_batch'], dtype=tf.float32)
            # xentropy = keras.losses.categorical_crossentropy(y_true, y_pred)
            # hsic_loss = -K.sum(x_kernel*Ku_batch)
            Ku_batch = tf.convert_to_tensor(db['Ku_batch'], dtype=tf.float32)
            xentropy = keras.losses.categorical_crossentropy(y_true, y_pred)
            H = tf.eye(batch_size) - tf.ones((batch_size,batch_size))/float(batch_size)
            x_kernel_H = tf.matmul(x_kernel, H)
            y_kernel_H = tf.matmul(Ku_batch, H)
            hsic_loss = -tf.linalg.trace(tf.matmul(x_kernel_H, y_kernel_H))
            return 0*xentropy+ hsic_loss
        return loss

    model_supervised.compile(loss=supervised_hsic_loss(lam=0),
                  optimizer=keras.optimizers.Adadelta(10),
                  metrics=['accuracy'])

    model_unsupervised.compile(loss=unsupervised_hsic_loss(lam=0),
                optimizer=keras.optimizers.Adadelta(10),
                metrics=['accuracy'])

    return model_xentropy, model_supervised, model_unsupervised, feature_extractor, kernel_checker, kernel_layer, ae

def get_expand_models(input_shape, old_model, num_new_classes, num_new_features=32):
    """
    Expand original classification model to accomodate novel classes
    """
    x_input = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
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

def show_and_save_metrics(exp_name, x, x_features, y_true, y_cluster, results, res_entry, args):
    acc_train = cluster_accuracy(y_true, y_cluster)
    # ARI_ex = metrics.adjusted_rand_score(labels_true_ex, labels_cluster_ex)
    # NMI_ex = metrics.adjusted_mutual_info_score(labels_true_ex, labels_cluster_ex)
    # HS_ex = metrics.homogeneity_score(labels_true_ex, labels_cluster_ex)
    # VM_ex = metrics.v_measure_score(labels_true_ex, labels_cluster_ex)
    ARI_train = metrics.adjusted_rand_score(y_true, y_cluster)
    NMI_train = metrics.adjusted_mutual_info_score(y_true, y_cluster)
    HS_train = metrics.homogeneity_score(y_true, y_cluster)
    VM_train = metrics.v_measure_score(y_true, y_cluster)
    AMI_train = metrics.adjusted_mutual_info_score(y_true, y_cluster, average_method='arithmetic')
    HCV_train = metrics.homogeneity_completeness_v_measure(y_true, y_cluster)
    Sil_train = metrics.silhouette_score(x_features, y_cluster, metric='euclidean', sample_size=max(min(args.sample_size,len(x)), int(len(x)/10)) if args.sample_size is not None else None)
    # ContMat_train = metrics.cluster.contingency_matrix(y_true, y_cluster)
    print('\n' + exp_name + ':\n{0:>40s} {1:.4f}\n{2:>40s} {3:.4f}\n{4:>40s} {5:.4f}\n{6:>40s} {7:.4f}\n{8:>40s} {9:.4f}\n{10:>40s} [{11:.4f}, {12:.4f}, {13:.4f}]\n{14:>40s} {15:.4f}'.format("Adjusted Rand Index:", ARI_train, "Normalized Mutual Information:", NMI_train, "Homogeneity Score:", HS_train, "V measure:", VM_train, "Silhouette Coefficient:", Sil_train,"Homogeneity, Completeness, V-measure:", *HCV_train, "Cluster Accuracy:", acc_train))
    results[res_entry]['ARI'].append(ARI_train)
    results[res_entry]['AMI'].append(AMI_train)
    results[res_entry]['Silhouette'].append(Sil_train)
    results[res_entry]['Homogeneity'].append(HCV_train[0])
    results[res_entry]['Completeness'].append(HCV_train[1])
    results[res_entry]['V-measure'].append(HCV_train[2])
    results[res_entry]['Cluster Acc'].append(acc_train)

def main():
    parser = argparse.ArgumentParser(description="Detect New Classes", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-nc', '--new_classes', type=str, default='25',
                        help='List of MNIST digits to be considered as new classes. All other classes are considered as old')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
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

    parser.add_argument('--supervision', action="store_true")
    parser.add_argument('--knet_epochs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

    new_classes = list(set(int(s) for s in args.new_classes)) # [0,6]
    new_classes.sort()
    old_classes = list(x for x in range(NUM_CLASSES) if x not in new_classes)

    filename = 'MNIST_hsic_semi.pkl'
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
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif args.dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
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
    db['sigma'] = 1 # initial setup
    db['sig_ratio'] = args.sigma
    model, s_hsic_model, u_hsic_model, feature_extractor, kernel_checker, kernel_layer, ae = get_models(input_shape, len(old_classes), lam=args.lam, sigma=db['sigma'], batch_size=args.batch_size)

    # train classifier use normal cross entropy loss
    model.fit(x_train,
              labels_to_categorical(y_train, old_classes),
              batch_size=args.batch_size,
              epochs = args.epochs,
              verbose=1,
              validation_data=(x_val,
                               labels_to_categorical(y_val, old_classes)))
    score = model.evaluate(x_test,
                           labels_to_categorical(y_test, old_classes), batch_size=args.batch_size, verbose=0)
    print('\n'+'*'*75+'\n\n\nTest loss: {0:.4f}\nTest accuracy: {1:.4f}'.format(score[0], score[1]))
    results[res_entry]['Accuracy'].append(score[1])

    # train ae
    # ae.fit(x_test_new,
    #        x_test_new,
    #        batch_size=args.batch_size,
    #        epochs = 20,
    #        verbose=1)

    # Here we use part of the 
    # 1. old data for supervised hsic
    # 2. old + new data for unsupervised hsic

    # dataset subsampling
    x_sup_sub, y_sup_sub = subsample_classes(x_train, y_train, samples_per_class=args.batch_size)

    x_train_whole = np.concatenate([x_train, x_train_new])
    y_train_whole = np.concatenate([y_train, y_train_new])

    x_unsup_sub, y_unsup_sub = subsample_classes(x_train_whole, y_train_whole, samples_per_class=args.batch_size)

    # train the 2 models alternately, updating U at the end of every epoch

    for i in range(args.knet_epochs):
        if i == 0:
            # initialization of U etc.
            # TODO: Could initialize by latent feature as well!
            # Get the latent features!
            phi_x = feature_extractor.predict(x_unsup_sub)
            initialize_embedding(phi_x, y_unsup_sub, num_clusters=10, sig_ratio=db['sig_ratio'])
            update_Ku()
            num_batches = len(y_unsup_sub) / args.batch_size
            id_list = np.arange(len(y_unsup_sub))
            # now the sig has changed, sync to our network
            kernel_layer.set_sigma(db['sigma'])

        # supervised
        if args.supervision: 
            s_hsic_model.fit(
                x_sup_sub,
                labels_to_categorical(y_sup_sub, old_classes),
                batch_size=args.batch_size
            )
        
        
        np.random.shuffle(id_list)
        # unsupervised
        for j in range(int(num_batches)):
            indices = id_list[j*args.batch_size:(j+1)*args.batch_size]
            # update Ku_batch as well
            update_Ku_batch(indices)
            x_batch = x_unsup_sub[indices]
            # give it dummy y as y is not needed in unsupervised case
            y_dummy_batch = np.ones(x_batch.shape[0])
            y_dummy_batch = labels_to_categorical(y_dummy_batch, old_classes)
            hsic_batch_loss = u_hsic_model.train_on_batch(x_batch, y_dummy_batch)
            print('Unsupervised hsic loss : {}'.format(hsic_batch_loss[0]))

        # Update U
        phi_x = feature_extractor.predict(x_unsup_sub)
        update_U(phi_x)
        # check performance on sub
        pred_y_unsup_sub = cluster.KMeans(n_clusters=NUM_CLASSES).fit_predict(phi_x)
        AMI_sub = metrics.adjusted_mutual_info_score(y_unsup_sub, pred_y_unsup_sub, average_method='arithmetic')
        acc_sub = cluster_accuracy(y_unsup_sub, pred_y_unsup_sub)
        print('HSIC Epoch {}: AMI: {:.3f}, ACC: {:.3f}'.format(i, AMI_sub, acc_sub))


    # # cluster training samples (old classes)
    # x_train_features = feature_extractor.predict(np.concatenate((x_train,x_val)))
    # kmeans_train = cluster.KMeans(n_clusters=len(old_classes)).fit(x_train_features)
    # y_train_cluster = kmeans_train.labels_
    # show_and_save_metrics(exp_name='Old train+val', x=np.concatenate((x_train,x_val)), x_features=x_train_features, 
    #                         y_true=np.concatenate((y_train,y_val)), y_cluster=y_train_cluster, results=results, res_entry=res_entry, args=args)

    # # cluster testing samples (old classes)
    # x_test_features = feature_extractor.predict(x_test)
    # kmeans_test = cluster.KMeans(n_clusters=len(old_classes)).fit(x_test_features)
    # y_test_cluster = kmeans_test.labels_
    # show_and_save_metrics(exp_name='Old test', x=x_test, x_features=x_test_features, 
    #                         y_true=y_test, y_cluster=y_test_cluster, results=results, res_entry=res_entry, args=args)

    # cluster training samples (new classes)
    x_train_new_features = feature_extractor.predict(x_train_new)
    kmeans_train_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_train_new_features)
    y_train_new_cluster = kmeans_train_new.labels_
    show_and_save_metrics(exp_name='New train, Before Expansion', x=x_train_new, x_features=x_train_new_features, 
                            y_true=y_train_new, y_cluster=y_train_new_cluster, results=results, res_entry=res_entry, args=args)

    

    ContMat_train_new = metrics.cluster.contingency_matrix(y_train_new, y_train_new_cluster)
    print('\nContingency matrix:')
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in ContMat_train_new]))
    print('\n\n'+'*'*75+'\n')

    # cluster testing samples (new classes)
    x_test_new_features = feature_extractor.predict(x_test_new)
    kmeans_test_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_test_new_features)
    y_test_new_cluster = kmeans_test_new.labels_
    show_and_save_metrics(exp_name='New test, Before Expansion', x=x_test_new, x_features=x_test_new_features, 
                            y_true=y_test_new, y_cluster=y_test_new_cluster, results=results, res_entry=res_entry, args=args)

    # save for tsne
    with open('tsne_utils/x_new_latent_be.pkl', 'wb') as f:
        pickle.dump(x_test_new_features, f)
    x_test_old_features = feature_extractor.predict(x_test)
    with open('tsne_utils/x_old_latent_be.pkl', 'wb') as f:
        pickle.dump(x_test_old_features, f)

    # save labels
    with open('tsne_utils/y_new.pkl', 'wb') as f:
        pickle.dump(y_test_new, f)
    with open('tsne_utils/y_old.pkl', 'wb') as f:
        pickle.dump(y_test, f)

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
    model_new, feature_extractor_new = get_expand_models(input_shape, model, len(new_classes))
    model_new.fit(np.concatenate((x_train,x_train_new)),
                  keras.utils.to_categorical(np.concatenate((y_train,y_train_new_pred)), NUM_CLASSES),
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=1,
                  validation_data=(np.concatenate((x_val,x_val_new)),
                                   keras.utils.to_categorical(np.concatenate((y_val,y_val_new_pred)), NUM_CLASSES)))
    # check classification acc
    score_new = model_new.evaluate(np.concatenate((x_test,x_test_new)),
                                   keras.utils.to_categorical(np.concatenate((y_test,y_test_new)), NUM_CLASSES),
                                   verbose=0)
    print('\n'+'*'*75+'\n\nAll classes\nTest loss: {0:.4f}\nTest accuracy: {1:.4f}'.format(score_new[0], score_new[1]))
    results[res_entry]['Accuracy'].append(score_new[1])
    # check old classification acc
    score_new_old = model_new.evaluate(x_test,
                                   keras.utils.to_categorical(y_test, NUM_CLASSES),
                                   verbose=0)
    print('\n'+'*'*75+'\n\Old classes\nTest loss: {0:.4f}\nTest accuracy: {1:.4f}'.format(score_new_old[0], score_new_old[1]))
    # check new classification acc
    score_new_new = model_new.evaluate(x_test_new,
                                   keras.utils.to_categorical(y_test_new, NUM_CLASSES),
                                   verbose=0)
    print('\n'+'*'*75+'\n\nAll classes\nTest loss: {0:.4f}\nTest accuracy: {1:.4f}'.format(score_new_new[0], score_new_new[1]))


    ConfMat_train = metrics.confusion_matrix(np.concatenate((y_train,y_val,y_train_new,y_val_new)),
                                             np.argmax(model_new.predict(np.concatenate((x_train,x_val,x_train_new,x_val_new))), axis=1))
    print('\nConfusion matrix (all train+val):')
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in ConfMat_train]))

    ConfMat_test = metrics.confusion_matrix(np.concatenate((y_test,y_test_new)),
                                                np.argmax(model_new.predict(np.concatenate((x_test,x_test_new))), axis=1))
    print('\nConfusion matrix (all test):')
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in ConfMat_test]))


    # # cluster training samples (old classes)
    # x_train_features_old = feature_extractor.predict(np.concatenate((x_train,x_val)))
    # x_train_features_new = feature_extractor_new.predict(np.concatenate((x_train,x_val)))
    # x_train_features = np.concatenate((x_train_features_old,x_train_features_new), axis=1)
    # kmeans_train = cluster.KMeans(n_clusters=len(old_classes)).fit(x_train_features)
    # y_train_cluster = kmeans_train.labels_
    # show_and_save_metrics(exp_name='Old train+val', x=np.concatenate((x_train,x_val)), x_features=x_train_features, 
    #                         y_true=np.concatenate((y_train,y_val)), y_cluster=y_train_cluster, results=results, res_entry=res_entry, args=args)

    # # cluster testing samples (old classes)
    x_test_features_old = feature_extractor.predict(x_test)
    x_test_features_new = feature_extractor_new.predict(x_test)
    x_test_old_features = np.concatenate((x_test_features_old,x_test_features_new), axis=1)
    # kmeans_test = cluster.KMeans(n_clusters=len(old_classes)).fit(x_test_features)
    # y_test_cluster = kmeans_test.labels_
    # show_and_save_metrics(exp_name='Old test', x=x_test, x_features=x_test_features, 
    #                         y_true=y_test, y_cluster=y_test_cluster, results=results, res_entry=res_entry, args=args)

    # # cluster training samples (new classes)
    # x_train_new_features_old = feature_extractor.predict(x_train_new)
    # x_train_new_features_new = feature_extractor_new.predict(x_train_new)
    # x_train_new_features = np.concatenate((x_train_new_features_old,x_train_new_features_new), axis=1)
    # kmeans_train_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_train_new_features)
    # y_train_new_cluster = kmeans_train_new.labels_
    # show_and_save_metrics(exp_name='New train', x=x_train_new, x_features=x_train_new_features, 
    #                         y_true=y_train_new, y_cluster=y_train_new_cluster, results=results, res_entry=res_entry, args=args)

    # cluster testing samples (new classes)
    x_test_new_features_old = feature_extractor.predict(x_test_new)
    x_test_new_features_new = feature_extractor_new.predict(x_test_new)
    x_test_new_features = np.concatenate((x_test_new_features_old,x_test_new_features_new), axis=1)
    kmeans_test_new = cluster.KMeans(n_clusters=len(new_classes)).fit(x_test_new_features)
    y_test_new_cluster = kmeans_test_new.labels_
    show_and_save_metrics(exp_name='New test', x=x_test_new, x_features=x_test_new_features, 
                            y_true=y_test_new, y_cluster=y_test_new_cluster, results=results, res_entry=res_entry, args=args)
    
     # save for tsne
    with open('tsne_utils/x_new_latent_ae.pkl', 'wb') as f:
        pickle.dump(x_test_new_features, f)
    with open('tsne_utils/x_old_latent_ae.pkl', 'wb') as f:
        pickle.dump(x_test_old_features, f)



    with open(args.save_path + filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test_initialize_embedding():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()   
    x_train = x_train.reshape(x_train.shape[0], -1) 
    x_train, y_train = subsample_classes(x_train, y_train, samples_per_class=10)
    x_train = x_train / 255.0
    initialize_embedding(x_train, y_train, num_clusters=10, sig_ratio=0.4)
    print(y_train)


if __name__ == '__main__':
    main()
    # test_initialize_embedding()
    

"""
#    TODO: Get class priors by feeding x_test, y_test to trained feature_extractor
"""

