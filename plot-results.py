#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:38:05 2020

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import argparse
import pickle
import numpy as np

from matplotlib import rcParams, cm
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'cmr10' # 'STIXGeneral' for accented characters
rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Plot Results for New Class Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--resultspath', type=str,
                        default='/Users/agritsenko/Dropbox/Research/NEU-SPIRAL/NewClassDetection/MNIST.pkl',
                        help='Path to directory with results')
    parser.add_argument('--epochs', type=str, default='10',
                        help='Specify # epochs to load results')
    
    args = parser.parse_args()
    
    data = pickle.load(open(args.resultspath, 'rb'))
    
    metrics = {'Accuracy':[], 'AMI':[]}
    keys = []
    
    for key in list(data.keys()):
        if key[key.find('-')+1:]==args.epochs:
            keys.append(key[:key.find('-')])
            for metric in metrics:
                metrics[metric].append(data[key][metric])
    
    for metric in metrics:
        metrics[metric] = np.array(metrics[metric])
    
    
    colors = []
    n_cols = 6 #len(metrics) * 2
    for i in range(n_cols):
        if i<n_cols/2:
            colors.append(cm.plasma(i / (n_cols/2)))
        else:
            colors.append(cm.viridis_r(i / (n_cols/2) - 1))
    colors = [colors[i*2] for i in range(int(n_cols/2))] + \
             [colors[i*2+1] for i in range(int(n_cols/2))]
#    colors = colors[2:] + colors[:2]

    fontsize = 36
    markersize = 18
    figsizescaler = 1.7
    fig = plt.figure(figsize = (figsizescaler*(len(keys)), 10))
    ax = fig.add_subplot(1,1,1)
    
    transparency = [1,1,1,\
                    0,1,1]
#    plot = 0
#    for metric in metrics:
#        ax.plot(metrics[metric][:,0], label=metric+'-Old',
#                color=colors[plot], marker='X', markersize=markersize, linestyle='', alpha=transparency[plot])
#        plot += 2
#    plot = 1
#    for metric in metrics:
#        ax.plot(metrics[metric][:,int(metrics[metric].shape[1]/2)], 
#                label=metric+('-New' if metrics[metric].shape[1]==4 else '-Old+New'),
#                color=colors[plot], marker='*', markersize=markersize, linestyle='', alpha=transparency[plot])
#        plot += 2
    ax.plot(metrics['AMI'][:,0], label='AMI-Old',
            color=colors[2], marker='X', markersize=markersize, linestyle='', alpha=transparency[1])
    ax.plot(metrics['AMI'][:,2], label='AMI-New',
            color=colors[4], marker='X', markersize=markersize, linestyle='', alpha=transparency[4])
    ax.plot(metrics['AMI'][:,5], label='AMI-Old (upd.)',
            color=colors[3], marker='P', markersize=markersize, linestyle='', alpha=transparency[2])
    ax.plot(metrics['AMI'][:,7], label='AMI-New (upd.)',
            color=colors[5], marker='P', markersize=markersize, linestyle='', alpha=transparency[5])
    ax.plot(metrics['Accuracy'][:,0], label='Accuracy-Old',
            color=colors[0], marker='*', markersize=markersize, linestyle='', alpha=transparency[0])
    ax.plot(metrics['Accuracy'][:,1], label='Accuracy-Old+New',
            color=colors[1], marker='*', markersize=markersize, linestyle='', alpha=transparency[3])
    
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(map(lambda x:'['+"".join([s+',' for s in x])[:-1]+']',keys), fontsize=fontsize)
    ax.set_xlim(-.1,len(keys)-.9)
    ax.set_ylim(0.17030481097700517, 1.0339219931280792)
    ax.tick_params(axis='y',labelsize=fontsize)
    ax.grid(b=True,linestyle=':')
    
    plot_order = [4,5,0,1,2,3]
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='lower right')
    ax.legend([handles[i] for i in plot_order], [labels[i] for i in plot_order],
              edgecolor='white', fontsize=fontsize,
              loc=8, bbox_to_anchor=(0, -0.27, 1, .102),
              mode='expand',
              ncol=3,#len(metrics),
              columnspacing=0.2,
              borderaxespad=-0.2, borderpad=0.2,
              handlelength=1.0, handletextpad=0.1)
    
    
    
    
    