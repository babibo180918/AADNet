import os
import sys
import traceback
from os.path import isfile
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import mne

import scipy
import numpy as np
from mesd_toolbox.mesd_toolbox import *

plt.rcParams["font.family"] = "Times New Roman"
#colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan', 'tab:olive']
colors = ['tab:gray', 'tab:blue', 'tab:brown', 'tab:red', 'lime', 'tab:purple', 'tab:olive', 'tab:red', 'tab:orange', 'tab:pink']
markers = ['o', '^', '*', 's', 'x', 'p', '1', '+', '<', 'x']
titlefontsizebig = 22
labelfontsizebig = 22
tickfontsizebig = 20
legendfontsizebig = 16
textfontsizebig = 20
#
titlefontsize = 16
labelfontsize = 16
tickfontsize = 14
legendfontsize = 12
textfontsize = 12
# figure size (mm)
fig_w_big = 180
fig_h_big = 120
fig_w_small = 180
fig_h_small = 120

def one_hot_encode(numOfClasses, labels):
    encoded_labels = np.zeros((numOfClasses, len(labels)), dtype=int)
    for i in range(0, len(labels)):
        if labels[i] > 0:
            encoded_labels[labels[i]-1, i] = 1
            
    return encoded_labels

def getFileList(in_path):
    filepaths = []
    if os.path.isfile(in_path):
        filepaths.append(in_path)
    elif os.path.isdir(in_path):
        for filename in glob.glob(in_path + '/**/*.*', recursive=True):
            filepaths.append(filename)
    else:
        print("Path is invalid: " + in_path)
        return None

    return filepaths

def addToHDF5(filepath, *args):
    numOfArgs = len(args)
    if numOfArgs%2 != 0:
        print("Number of arguments are incorrect.")
        return

def print_memory_info(device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    print(f'total memory: {t/1024/1024/1024}')
    print(f'reserved memory: {r/1024/1024/1024}')
    print(f'allocated memory: {a/1024/1024/1024}')
    print(f'free memory: {f/1024/1024/1024}')

def accuracy(scores, yb, thresh=0.5):
    if scores.dim() > 1:
        score2prob = nn.Softmax(dim=-1)
        preds = torch.argmax(score2prob(scores), dim=-1)
        acc = (preds == yb).float().mean()
    else:
        preds = scores>thresh
        acc = (preds == yb).float().mean()
    return acc    
    
def metrics(y_hat,y_true, thresh=None, weighted=False):
    if (thresh is None):
        max_acc = 0
        for i in range(100):
            thr = 0.5 + ((-1)**i)*0.01*int((i+1)/2)
            y_pred = y_hat>thr
            true_pos = torch.count_nonzero((y_true==1)&(y_pred==1))
            false_pos = torch.count_nonzero((y_true==0)&(y_pred==1))
            true_neg = torch.count_nonzero((y_true==0)&(y_pred==0))
            false_neg = torch.count_nonzero((y_true==1)&(y_pred==0))
            acc = (true_pos+true_neg)/len(y_hat)
            if acc>max_acc:
                max_acc = acc
                thresh = thr
    y_pred = y_hat>thresh
    true_pos = torch.count_nonzero((y_true==1)&(y_pred==1))
    false_pos = torch.count_nonzero((y_true==0)&(y_pred==1))
    true_neg = torch.count_nonzero((y_true==0)&(y_pred==0))
    false_neg = torch.count_nonzero((y_true==1)&(y_pred==0))
    acc = torch.count_nonzero(y_true==y_pred)/torch.numel(y_pred)
    if weighted:    
        classes, weights = torch.unique(y_true, return_counts=True)
        n_classes = len(classes)
        weights = weights.float()
        for i in range(n_classes):
            weights[i] = torch.count_nonzero((y_true==y_pred)&(y_pred==classes[i])).float()/weights[i]
        acc = weights.mean()
    return (true_pos, false_pos, true_neg, false_neg, acc, thresh)
    
def multiclass_accuracy(scores, yb):
    score2prob = nn.Softmax(dim=1)
    preds = torch.argmax(score2prob(scores), dim=1)
    return (preds == yb).float().mean()   

def pearson_scorer(estimator, X, y):
    assert X.ndim >= 2, "the dimension of X must be greater than or equal to 2."
    X_shape = X.shape
    y_pred = estimator.predict(X.reshape(-1,X_shape[-1])).reshape(X_shape[:-1])
    shape = y_pred.shape
    y = y.reshape(-1,shape[-1])
    y_pred = y_pred.reshape(-1,shape[-1])
    return np.diag(np.corrcoef(y, y_pred), y.shape[-2]).reshape(shape[:-1])

def getHardLabels(X, y, n_classes):
    idxs = (y==0)
    for i in range(1,n_classes):
        idxs = idxs|(y==i)
    return X[idxs], y[idxs]
           
def plot_compare_bar(compare_data, bar_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None):
    nBars = len(compare_data)
    width = 0.2
    plt.clf()
    for i in range(nBars):
        (nXticks,) = compare_data[i].shape
        loc = np.arange(nXticks) - (nBars-1)*width/2 + i*width
        rects = plt.bar(loc, compare_data[i], width, label=bar_labels[i], color=colors[i], zorder=3)
    #
    ticks = np.arange(len(xtick_labels))  # the label locations
    plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    # plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['${}^*$0.5', 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.ylim(0.5, 1)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=labelfontsize)      
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    plt.legend(fontsize=legendfontsize)
    plt.grid(axis = 'y', linestyle='--', linewidth=1.0, zorder=0)

    if save_path is not None:
        plt.gcf().set_size_inches(fig_w_small/25.4, fig_h_small/25.4)    
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
def plot_compare_bar_withSTDbar(compare_data, bar_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None):
    nBars = len(compare_data)
    width = 0.2
    plt.clf()
    ticks = np.arange(len(xtick_labels))  # the label locations
    plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    # plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['${}^*$0.5', 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.ylim(0.5, 1)    
    plt.grid(axis = 'y', linestyle='--', linewidth=1.0, zorder=0)
    for i in range(nBars):
        (nXticks, obvs) = compare_data[i].shape
        mean = np.mean(compare_data[i], 1, keepdims=False)
        std = np.std(compare_data[i], 1, keepdims=False)
        loc = np.arange(nXticks) - (nBars-1)*width/2 + i*width
        rects = plt.bar(loc, mean, width, label=bar_labels[i], color=colors[i], zorder=3)
        plt.errorbar(loc, mean, std, capsize=3, linestyle='none', color='k', zorder=3)
    #
    if x_label is not None:
        plt.xlabel(x_label, fontsize=labelfontsize)      
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    plt.legend(fontsize=legendfontsize)
    
    if save_path is not None:
        plt.gcf().set_size_inches(fig_w_small/25.4, fig_h_small/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_compare_line(compare_data, bar_labels=None, xtick_labels=None, x_label=None, y_label=None, title=None, save_path=None):
    (nBars, nTicks) = compare_data.shape
    #
    x = np.arange(nTicks)  # the label locations
    width = 0.2
    plt.clf()
    fig, ax = plt.subplots()
    for i in range(nBars):
        rects = ax.bar(x - (nBars-1)*width/2+i*width, compare_data[i], width, label=bar_labels[i])
        ax.bar_label(rects, padding=2, fontsize=6)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=labelfontsize)      
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(x)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.legend(loc='lower right')
    #autolabel(ax, rects1)
    #autolabel(ax, rects2)
    fig.tight_layout()  
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')          

def plot_compare_model_AVG_withSTDbar(train, test, names=None, x_label=None, y_label=None, title=None, save_path=None):
    nModels = len(train)
    #
    x = np.arange(nModels)  # the label locations
    width = 0.2
    plt.clf()
    plt.errorbar(x - width/2, np.mean(train, 1, keepdims=False), np.std(train, 1, keepdims=False), capsize=3, linestyle='none', marker='o', label='train')
    plt.errorbar(x + width/2, np.mean(test, 1, keepdims=False), np.std(test, 1, keepdims=False), capsize=3, linestyle='none', marker='x', label='test')
    if x_label is not None:
        plt.xlabel(x_label, fontsize=labelfontsize)      
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    plt.xticks(x)
    if names is not None:
        plt.xticks(ticks=x, labels=names, fontsize=tickfontsize)
    plt.legend(fontsize=legendfontsize)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
def plot_line_barSTD(compare_data, line_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None, fig=None):
    nLines = len(compare_data)
    width = 0.1
    if fig is None:
        plt.clf()
    for i in range(nLines):
        (nXticks, obvs) = compare_data[i].shape
        line_mean = np.mean(compare_data[i], 1, keepdims=False)
        line_std = np.std(compare_data[i], 1, keepdims=False)
        # loc = np.arange(nXticks) - (nLines-1)*width/2 + i*width
        loc = xtick_labels - (nLines-1)*width/2 + i*width
        # plt.plot(loc, line_mean, '-', marker=markers[i], color=colors[i], label=line_labels[i])
        plt.plot(loc, line_mean, '-', color=colors[i], label=line_labels[i])
        for j in range(nXticks):
            # plt.errorbar(j - (nLines-1)*width/2 + i*width, line_mean[j], line_std[j], capsize=3, linestyle='none', color=colors[i])
            plt.errorbar(loc[j], line_mean[j], line_std[j], capsize=3, linestyle='none', color=colors[i])
    #
    ticks = np.arange(len(xtick_labels))  # the label locations
    # plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    plt.xticks(ticks=xtick_labels, fontsize=tickfontsize)
    # plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['${}^*$0.5', 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.ylim(0.5, 1.0)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=labelfontsize)    
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    # plt.legend(fontsize=legendfontsize, ncol=4)
    plt.legend(loc='upper left', fontsize=legendfontsize, ncol=5, columnspacing=1.0)
    plt.grid(axis = 'y', linestyle='--', linewidth=1.0, zorder=3)
    fig = plt.gcf()
    
    if save_path is not None:
        fig.set_size_inches(fig_w_small/25.4, fig_h_small/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_line_filledSTD(y, line_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None, fig=None):
    nLines = len(y)
    plt.clf()
    for i in range(nLines):
        mu = y[i].mean(axis=1, keepdims=False)
        std = y[i].std(axis=1, keepdims=False)
        plt.plot(xtick_labels, mu, "-o", color=colors[i], linewidth=2.0, markersize=4, label=line_labels[i])
        plt.fill_between(xtick_labels, mu+0.5*std, mu-0.5*std, facecolor=colors[i], alpha=0.25)
    plt.xticks(ticks=xtick_labels, fontsize=tickfontsize)
    # plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['${}^*$0.5', 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.ylim(0.5, 1.0)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=labelfontsize)      
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    plt.legend(loc='upper left', fontsize=legendfontsize, ncol=5)
    
    if save_path is not None:
        plt.gcf().set_size_inches(fig_w_small/25.4, fig_h_small/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_box_barSTD(y, box_labels, xtick_labels, x_label=None, y_label=None, title=None, stats=None, chance_lv=None, save_path=None, fig=None):
    nLines = len(y)
    width = 0.1
    space = 0.02
    ystats = [0.42, 0.45, 0.48, 0.51, 0.54, 0.57]
    # boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
    flierprops = dict(marker='o', markersize=4.0, linestyle='none', markeredgewidth=0.5)
    medianprops = dict(linestyle='-', linewidth=0.5)
    meanprops  = dict(linestyle='-', linewidth=0.5, color='#00ff00')
    
    x_ticks = np.arange(len(xtick_labels))  # the tick locations
    #x_ticks = xtick_labels  # the tick locations    
    plt.clf()
    fig, ax = plt.subplots()
    bplots = []
    lplots = []
    line_labels = []
    for i in range(nLines):
        mu = y[i].mean(axis=1, keepdims=False)
        std = y[i].std(axis=1, keepdims=False)
        # loc = x_ticks - (nLines-1)*width/2 + i*width
        loc = x_ticks - (nLines-1)*(width+space)/2 + i*(width+space)
        bp = ax.boxplot(y[i].T, patch_artist=True, widths=width, positions=loc, whis=1.5, 
                        showmeans=True, meanline=True, showfliers=True, 
                        flierprops=flierprops, medianprops=medianprops, meanprops=meanprops)
        bplots.append(bp['boxes'][0])
        [bp['boxes'][j].set_facecolor(colors[i]) for j in range(len(loc))]
        
    lplots.append(bp['means'][0])
    line_labels.append('mean')
    lplots.append(bp['medians'][0])
    line_labels.append('median')
    ax.set_xticks(ticks=x_ticks, labels=xtick_labels, fontsize=tickfontsize)
    # ax.set_yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['${}^*$0.5', 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    ax.set_yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    ax.set_xlim(-0.5, len(xtick_labels)-0.5)
    ax.set_ylim(0.4, 1.0)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=labelfontsize)      
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        ax.set_title(title, fontsize=titlefontsize)
    if chance_lv is not None:
        cl_width = width*nLines + space*(nLines-1)
        from_x = x_ticks - cl_width/2
        to_x = x_ticks + cl_width/2
        for i in range(len(x_ticks)):
            pl = ax.plot([from_x[i], to_x[i]],[chance_lv[i], chance_lv[i]], '--', linewidth=0.5, color='gray')
        lplots.append(pl[0])
        line_labels.append('chance level')
    ax.legend(bplots+lplots, box_labels+line_labels, fontsize=legendfontsize, ncol=2, columnspacing=0.5, loc='upper left')
    if stats is not None:
        xloc2 = x_ticks - (nLines-1)*(width+space)/2 + (nLines-1)*(width+space) # location of proposed model
        for i in range(nLines-1):
            xloc = x_ticks - (nLines-1)*(width+space)/2 + i*(width+space)
            for j in range(len(xloc)):
                ax.plot([xloc[j],xloc2[j]], [ystats[i], ystats[i]], color='k', marker= "|", markersize= 3.0, linestyle='-', linewidth=0.5)
                if(stats[i][j] <= 0.05) and (stats[i][j] > 0.01):
                    text = '*'
                elif(stats[i][j] <= 0.01) and (stats[i][j] > 0.001):
                    text = '**'
                elif(stats[i][j] <= 0.001):
                    text = '***'
                else:
                    text = ''
                ax.text((xloc[j]+xloc2[j])/2, ystats[i]+0.001, text, horizontalalignment='center', verticalalignment='center')
        ax.set_ylim(0.4, 1.0)
        
    if save_path is not None:
        plt.gcf().set_size_inches(fig_w_small/25.4, fig_h_small/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()    

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_accuracy(valid_accs, test_accs, filename):
    valid_accs = np.array(valid_accs)
    test_accs = np.array(test_accs)
    avg_valid = valid_accs.mean()
    avg_test = test_accs.mean()
    valid_accs = np.append(valid_accs, [avg_valid], axis=0)
    test_accs = np.append(test_accs, [avg_test], axis=0)
    labels = []
    for i in range(len(valid_accs)):
        labels.append(f'{i}')
    labels[-1] = 'avg'
    #
    x = np.arange(len(valid_accs))  # the label locations
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, valid_accs, width, label=f'valid: {valid_accs[-1]:.3f}')
    rects2 = ax.bar(x + width/2, test_accs, width, label=f'test: {test_accs[-1]:.3f}')
    ax.set_ylabel('Accuracy')
    ax.set_title('Leave-one-out accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    #autolabel(ax, rects1)
    #autolabel(ax, rects2)
    fig.tight_layout()    
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def visualize(model, X, filename):
    print(model)
    model.to(X.device)
    g = make_dot(model(X).mean(), params=dict(model.named_parameters()))
    #g.render(filename, format="pdf", view=True, cleanup=True, quiet=True)
    
def permutation_test(x1, x2, n_iters:int=1000, tail=0, plot_hist=False, statistics='mean') -> np.ndarray:
    """ Perform permutation test of 2 random variables
    Arguments:
        x1: Variable 1 of type Numpy array corresponding to condition 1. The last dimension is applied for permutation test.
        x2: Variable 2 of type Numpy array corresponding to condition 2. The last dimension is applied for permutation test.
    Returns:
        The numpy array of p values 
    """
    N1 = x1.shape[-1]
    N2 = x2.shape[-1]
    x1 = np.swapaxes(x1, 0, -1)
    x2 = np.swapaxes(x2, 0, -1)
    if statistics=='median':
        T_obs = np.median(x1, axis=0, keepdims=True) - np.median(x2, axis=0, keepdims=True)
    else:
        T_obs = np.mean(x1, axis=0, keepdims=True) - np.mean(x2, axis=0, keepdims=True)
    shuffled = np.concatenate((x1, x2), axis=0)    
    H0 = np.zeros((n_iters,) + T_obs.shape)
    for i in np.arange(n_iters):
        np.random.shuffle(shuffled)
        if statistics=='median':
            H0[i] = np.median(shuffled[:N1,...], axis=0) - np.median(shuffled[N1:,...], axis=0)
        else:
            H0[i] = np.mean(shuffled[:N1,...], axis=0) - np.mean(shuffled[N1:,...], axis=0)
    if (tail==0):
        p_values = np.count_nonzero(np.abs(H0) >= np.abs(T_obs), axis=0)/n_iters
    elif (tail==-1):
        p_values = np.count_nonzero(H0 <= T_obs, axis=0)/n_iters
    elif (tail==1):
        p_values = np.count_nonzero(H0 >= T_obs, axis=0)/n_iters        
    
    p_values = np.swapaxes(p_values, 0, -1)
    if plot_hist:
        hist_data = H0.reshape(n_iters, -1)
        for i in range(hist_data.shape[1]):
            plt.figure(i)
            plt.hist(hist_data[:,i])
            plt.title(f'Sampling distribution of difference of means variables {i}')
            plt.show()
    return np.squeeze(p_values, axis=-1)    

def eog_reject_by_ICA(raw, picks):
    # eog events 
    try:
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name='EOG', event_id=998, picks=picks+['EOG'], tmin=-0.2, tmax=0.2, thresh = 100e-6)
        average_eog = eog_epochs.average()
        print('We found %i EOG events' % average_eog.nave)
        joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
        #average_eog.plot_joint(**joint_kwargs)
    except:
        pass
        
    method = 'fastica'
    # Choose other parameters
    n_components = 32  # if float, select n_components by explained variance of PCA
    decim = 3  # we need sufficient statistics, not all time points -> saves time
    random_state = 23
    ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state)
    
    ica.fit(raw, picks=picks, decim=decim)
    eog_inds, scores = ica.find_bads_eog(raw, ch_name='EOG', threshold=0.8, measure='correlation')
    if len(eog_inds) > 0:
        print(f'Bad EOG components: {eog_inds}')
        #ica.plot_scores(scores)
        #ica.plot_components()
        #ica.plot_overlay(raw, exclude=eog_inds)
        #ica.plot_properties(eog_epochs, picks=eog_inds)
    ica.exclude = eog_inds
    out_raw = raw.copy()
    out_raw.load_data()
    ica.apply(out_raw)
    
    return out_raw
    
def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.data}')
            
def plot_correlation(audio, eeg, corr, y):
    (batch_size, c, L) = eeg.shape
    print(f'label: {y}')
    for i in range(batch_size):
        print(f'corr0: {corr[i,0]}')
        print(f'corr1: {corr[i,1]}')
        for j in range(c):
            plt.clf()
            plt.plot(eeg[i,j], '-k', linewidth=1.0)
            plt.plot(audio[i,0], '-r', linewidth=2.0, label=f'chn0: {corr[i,0]}')
            plt.plot(audio[i,1], '-b', linewidth=2.0, label=f'chn0: {corr[i,1]}')
            plt.title(f'attended: {y[i]}')
            plt.legend()
            plt.show()     
            
def plot_correlation_score(scores, colunm_labels, row_labels, save_path=None):
    plt.clf()
    w = 0.25
    column_space = 0.05
    for i in range(len(row_labels)):
        for j in range(len(colunm_labels)):
            score = scores[i][j]
            x = j + (np.random.rand(len(score))-0.5)*w - (len(row_labels)-1)*(w+column_space)/2 + (w+column_space)*i
            if j==0:
                plt.plot(x, score, color=colors[i], marker='o', markersize=1, linestyle='', label=f'{row_labels[i]}')
            else:
                plt.plot(x, score, color=colors[i], marker='o', markersize=1, linestyle='')
    plt.xlabel('Decision window [s]', fontsize=labelfontsize)
    plt.ylabel('Pearson correlation', fontsize=labelfontsize)
    ticks = np.arange(len(colunm_labels))  # the label locations
    plt.xticks(ticks=ticks, labels=colunm_labels, fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')        
            
def plot_bitrate(num_class, acc, line_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None):    
    nLines = len(acc)    
    plt.clf()
    ticks = np.arange(len(xtick_labels))  # the tick locations
    # ticks = xtick_labels  # the tick locations
    for i in range(nLines):
        avg_acc = acc[i].mean(1, keepdims=False)
        br = (np.log2(num_class) + avg_acc*np.log2(avg_acc) + (1-avg_acc)*np.log2((1-avg_acc)/(num_class-1)))/np.array(xtick_labels)*60
        plt.plot(ticks, br, color=colors[i], marker='o', markersize=3, linestyle='-', label=f'{line_labels[i]}')
        
    plt.xlabel('Window length (s)', fontsize=labelfontsize)
    plt.ylabel('Bitrate (bit/min)', fontsize=labelfontsize)
    
    plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 1.0, 1.5, 2.0], fontsize=tickfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
        # plt.title(title, fontsize=titlefontsize, y=1.3)
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5)
    plt.legend(loc='upper center', fontsize=legendfontsize, ncol=5)
    if save_path is not None:
        plt.gcf().set_size_inches(fig_w_small/25.4, fig_h_small/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_esd(acc, line_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None):    
    nLines = len(acc)    
    plt.clf()
    ticks = np.arange(len(xtick_labels))  # the tick locations
    # ticks = xtick_labels  # the tick locations
    for i in range(nLines):
        avg_acc = acc[i].mean(1, keepdims=False)
        print(f'avg_acc: {avg_acc}')
        esd, *_ = compute_ESD(xtick_labels,avg_acc);
        plt.plot(ticks, esd, color=colors[i], marker='o', markersize=3, linestyle='-', label=f'{line_labels[i]}')
        
    plt.xlabel('Window length (s)', fontsize=labelfontsize)
    plt.ylabel('ESD (s)', fontsize=labelfontsize)
    
    plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    # plt.yticks(ticks=[0.5, 1.0, 1.5, 2.0], fontsize=tickfontsize)
    # plt.ylim(30, 180)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
        # plt.title(title, fontsize=titlefontsize, y=1.3)
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5)
    plt.legend(loc='upper center', fontsize=legendfontsize, ncol=5)
    if save_path is not None:
        plt.gcf().set_size_inches(fig_w_small/25.4, fig_h_small/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_mesd_hori(acc, line_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None):    
    nLines = len(acc)    
    plt.clf()
    fig, ax = plt.subplots()
    mesds = []
    flierprops = dict(marker='o', markersize=4.0, linestyle='none', markeredgewidth=0.5)
    medianprops = dict(linestyle='-', linewidth=1.0, color='k')
    whiskerprops = dict(linestyle='')
    max_mesd = 100
    for i in range(nLines):
        (nWd, nSbjs) = acc[i].shape
        mesd = np.zeros(nSbjs)
        acc[i][acc[i]<0.5] = 0.5
        for j in range(nSbjs):
            try:
                mesd[j], *_ = compute_MESD(xtick_labels, acc[i][:,j], N_min=5, P0=0.8, c=0.65)
            except AssertionError:
                _, _, tb = sys.exc_info()
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]            
                print(f'got AssertionError: {text}')
                continue
        mesd = np.delete(mesd, np.argwhere(mesd==0.0))
        bp = ax.boxplot(mesd, positions=[i],vert=False, patch_artist=True, whis=1.5, 
                        showmeans=False, meanline=False, showfliers=False, showbox=False, showcaps=False, 
                        flierprops=flierprops, medianprops=medianprops, whiskerprops=whiskerprops)
        points = mesd[mesd<max_mesd]
        outliers = mesd[mesd>max_mesd]        
        ax.plot(points, np.full(len(points), i), '.', markersize=8.0, color=colors[i])
        for line in bp['medians']:
            x, y = line.get_xydata()[1] # top of median line bbox=dict(facecolor='pink', alpha=0.5)
            ax.text(x, y+0.05, '%.1f s' % x, horizontalalignment='center', fontsize=textfontsize)
        if(len(outliers) > 0):
            ax.text(max_mesd, i, '(+%d)' % len(outliers), horizontalalignment='left', verticalalignment='center', fontsize=textfontsize, color=colors[i])
        ax.text(max_mesd+10, i, f'{line_labels[i]}', horizontalalignment='left', verticalalignment='center', fontsize=textfontsize, color=colors[i])

    ax.set_xticks([0, 50, max_mesd], labels=[0, 50, max_mesd], fontsize=tickfontsize)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) 
    ax.set_xlabel('MESD (s)', fontsize=labelfontsize)    
    ax.set_xlim([0,max_mesd])
    if title is not None:
        ax.set_title(title, fontsize=titlefontsize)
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5)
    # plt.legend(loc='upper center', fontsize=legendfontsize, ncol=5)
    if save_path is not None:
        plt.gcf().set_size_inches(fig_w_small/25.4, 90/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  

def plot_mesd(acc, line_labels, xtick_labels, x_label=None, y_label=None, title=None, save_path=None, stats=None):    
    # ystats = [-16, -8, 0, 8]
    ystats = [-15, -10, -5, 0, 5, 10]
    nLines = len(acc)    
    plt.clf()
    fig, ax = plt.subplots()
    mesds = []
    flierprops = dict(marker='o', markersize=4.0, linestyle='none', markeredgewidth=0.5)
    medianprops = dict(linestyle='-', linewidth=1.0, color='k')
    whiskerprops = dict(linestyle='')
    max_mesd = 100
    x_ticks = np.arange(len(line_labels))
    for i in range(nLines):
        (nWd, nSbjs) = acc[i].shape
        mesd = np.zeros(nSbjs)
        acc[i][acc[i]<0.5] = 0.5
        for j in range(nSbjs):
            try:
                mesd[j], *_ = compute_MESD(xtick_labels, acc[i][:,j], N_min=5, P0=0.8, c=0.65)
            except AssertionError:
                _, _, tb = sys.exc_info()
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]            
                print(f'got AssertionError: {text}')
                continue
        mesd = np.delete(mesd, np.argwhere(mesd==0.0))
        bp = ax.boxplot(mesd, positions=[i],vert=True, patch_artist=True, whis=1.5, 
                        showmeans=False, meanline=False, showfliers=False, showbox=False, showcaps=False, 
                        flierprops=flierprops, medianprops=medianprops, whiskerprops=whiskerprops)
        points = mesd[mesd<max_mesd]
        outliers = mesd[mesd>max_mesd]        
        ax.plot(np.full(len(points), i), points, '.', markersize=8.0, color=colors[i])
        for line in bp['medians']:
            x, y = line.get_xydata()[1] # top of median line bbox=dict(facecolor='pink', alpha=0.5)
            ax.text(x+0.1, y, '%.1f s' % y, horizontalalignment='left', verticalalignment='center', fontsize=textfontsize)
        if(len(outliers) > 0):
            ax.text(i, max_mesd, '(+%d)' % len(outliers), horizontalalignment='center', verticalalignment='center', fontsize=textfontsize, color=colors[i])
        # ax.text(max_mesd+10, i, f'{line_labels[i]}', horizontalalignment='left', verticalalignment='center', fontsize=textfontsize, color=colors[i])

    ax.set_yticks([0, 50, max_mesd], labels=[0, 50, max_mesd], fontsize=tickfontsize)
    ax.set_xticks(x_ticks, labels=line_labels, fontsize=tickfontsize)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False) 
    ax.set_ylabel('MESD (s)', fontsize=labelfontsize)    
    ax.set_ylim([-25,max_mesd+10])
    if title is not None:
        ax.set_title(title, fontsize=titlefontsize)

    if stats is not None:
        xloc2 = x_ticks[-1] # location of proposed model
        for i in range(nLines-1):
            ax.plot([x_ticks[i],x_ticks[-1]], [ystats[i], ystats[i]], color='k', marker= "|", markersize= 3.0, linestyle='-', linewidth=0.5)
            if(stats[i] <= 0.05) and (stats[i] > 0.01):
                text = '*'
            elif(stats[i] <= 0.01) and (stats[i] > 0.001):
                text = '**'
            elif(stats[i] <= 0.001):
                text = '***'
            else:
                text = ''
            ax.text((x_ticks[i]+x_ticks[-1])/2, ystats[i]+0.001, text, horizontalalignment='center', verticalalignment='center')
        ax.set_ylim([-25,max_mesd+10])        
        
    if save_path is not None:
        plt.gcf().set_size_inches(135/25.4, fig_h_small/25.4)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')          

    