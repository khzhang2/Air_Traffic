import numpy as np
import os
import random
import torch


def mape_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(eliminate_nan(np.fabs(labels[mask]-preds[mask])/labels[mask]))


def smape_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(2*np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))


def mae_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(np.fabs((labels[mask]-preds[mask])))


def nrmse_loss_func(preds, labels, m):
    mask = preds > m
    return np.sqrt(np.sum((preds[mask] - labels[mask])**2)/preds[mask].flatten().shape[0])/(labels[mask].max() - labels[mask].min())


def nmae_loss_func(preds, labels, m):
    mask = preds > m
    return np.mean(np.fabs((labels[mask]-preds[mask]))) / (labels[mask].max() - labels[mask].min())


def eliminate_nan(b):
    a = np.array(b)
    c = a[~np.isnan(a)]
    c = c[~np.isinf(c)]
    return c


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def save_model(net, name):
    num_fold = get_num_fold() + 1
    try:
        torch.save(net.state_dict(), './runs/run%i/%s.pth'%(num_fold, name))
    except:
        raise RuntimeError('No fold for this experiment created')


def get_num_fold():
    num_fold = len(next(iter(os.walk('./runs/')))[1])
    return num_fold

def get_CPC(pred, labels):
    res_min = np.concatenate([pred, labels], axis=1).min(axis=1).flatten()
    CPC = np.sum(res_min)*2 / (np.sum(pred) + np.sum(labels))
    return CPC
