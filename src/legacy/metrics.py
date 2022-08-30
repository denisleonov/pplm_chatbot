import torch
import numpy as np
from rouge import Rouge
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from copy import copy
from src.utils.utils import to_numpy

ROUGE_TYPES = ('rouge-1', 'rouge-2', 'rouge-L', 'rouge-mean')
LAST_ROUGES_SIZE = 100

epsilon = 1e-7

rouge = Rouge()


def get_rouges(pred_responses, gold_responses):
    """Function to get rouge scores of responces.
    :return list of ROUGE (1,2,l) and mean of this scores

    ROUGE scores are computed for every pair of predicted and gold sentences. 
    We use only F-measure, that means "weighted measure" of precision and recall in terms of n-gramms.
    
    """
    r = rouge.get_scores(pred_responses, gold_responses, avg=True)
    r = [r[f'rouge-{x}']['f'] for x in ('1', '2', 'l')]
    r += [np.mean(r)]

    return r

def get_scores(y_pred, y_true, average, multi_class):
    np_pred = to_numpy(y_pred)
    np_true = to_numpy(y_true)
    if multi_class != 'no':
        metrics = roc_auc_score(np_true, np_pred, average=average, multi_class=multi_class)
    else:
        metrics = roc_auc_score(np_true, np_pred, average)
        print('precision', average_precision_score(np_true, np_pred, average))
    return metrics

def get_precision(tp=None, tn=None, fp=None):
    """ Function to get positive predictive value (tp/(tp+fp)"""
    return tp / (tp + fp + epsilon)

def get_recall(tp=None, fn=None):
    """Function to get sensitivity (tp/p)"""
    return tp / (tp + fn + epsilon)

def get_tnr(tn=None, fp=None):
    """Function to get selectivity (tn/n)"""
    return tn / (tn + fp + epsilon)
def get_fnr(self, tp=None, fn=None):
    """Function to get miss rate (fn/p)"""
    return fn / (tp + fn + self.epsilon)

def get_fpr(self, tn=None, fp=None, cached=True):
    """Function to get false positive rate (fp/n)"""
    if cached:
        return self.fp / (self.tn + self.fp + self.epsilon)
    else:
        return fp / (tn + fp + self.epsilon)


def get_confusion(y_pred, y_true, device='cuda', labels=None):
    if y_true.dim() < 2:
        return get_sklearn_confusion(y_pred, y_true, device=device, labels=labels)
    else:
        tp = (y_true * y_pred).sum(dim=0).to(torch.int)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.int)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.int)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.int)
        return tp, tn, fp, fn

def to_one_hot(y, num_classes=6, device='cuda'):
    ones = torch.eye(num_classes).to(device)
    return ones[y].squeeze()

def get_sklearn_confusion(y_pred,y_true, device='cuda', labels=None):
    conf_mat = confusion_matrix(y_true.squeeze().cpu().numpy(),
                                y_pred.squeeze().cpu().numpy(),
                                labels=labels)
    tp = conf_mat.diagonal()
    conf_mat = np.array(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    tn = - (tp + fn + fp - conf_mat.sum())
    return torch.tensor(tp, device=device, dtype=torch.int), torch.tensor(tn, device=device, dtype=torch.int), \
           torch.tensor(fp, device=device, dtype=torch.int), torch.tensor(fn, device=device, dtype=torch.int)


def get_f(tp, tn, fp, fn, beta=1):
    precision = get_precision(tp, tn, fp)
    recall = get_recall(tp, fn)
    f1 = (1+(beta**2)) * (precision*recall) /   \
                                  ((beta**2)*precision + recall + epsilon)
    f1.clamp(min=epsilon, max=1-epsilon)
    return to_numpy(f1.mean())
