import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef, average_precision_score,
                             confusion_matrix, balanced_accuracy_score, accuracy_score)


def cm_score(cfs_mtx):
    """Infer scores from confusion matrix. Implemented for using with 'compute_metrics_core'"""
    tn, fp, fn, tp = cfs_mtx.ravel()

    def sen(*args):
        return tp / (tp + fn)  # sensitivity

    def spe(*args):
        return tn / (tn + fp)  # specificity

    def pre(*args):
        return tp / (tp + fp)  # precision

    def acc(*args):
        return (tp + tn) / (tp + tn + fp + fn)  # accuracy

    def wrapper(metric_name):
        if metric_name == 'sen':
            return sen
        elif metric_name == 'spe':
            return spe
        elif metric_name == 'pre':
            return pre
        elif metric_name == 'acc':
            return acc

    return wrapper


def get_metrics(cfs_mtx=None):
    metrics = {
        'auc': roc_auc_score,
        'auprc': average_precision_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef,
        'acc_b': balanced_accuracy_score,
        'acc': accuracy_score,
    }
    if cfs_mtx is not None:
        for k in ['sen', 'spe', 'pre', 'acc']:
            metrics.update({k: cfs_mtx(k)})
    return metrics


def compute_metrics_signals(labels_one_hot, predictions_one_hot,
                            metric_list=('auc', 'auprc', 'f1', 'mcc', 'acc', 'acc_b'),
                            current_epoch=None, verbose=True):
    labels, predictions = labels_one_hot.argmax(1), predictions_one_hot.argmax(1)
    cm = confusion_matrix(labels, predictions)
    metrics = get_metrics()
    scores = {}
    for metric in metric_list:
        l, p = (labels, predictions) if 'au' not in metric else (labels_one_hot, predictions_one_hot)
        if metric in ['acc', 'mcc', 'acc_b']:
            scores[metric] = metrics[metric](l, p)
        else:
            scores[metric] = metrics[metric](l, p, average='weighted')

    if verbose:
        df = pd.DataFrame([scores.values()], columns=[_.upper() for _ in scores.keys()],
                          index=[f'Epoch = {current_epoch if current_epoch else "Best"}'])
        print(df.round(3))
        if scores['acc_b'] > .6:
            print(cm)
    return scores


def compute_metrics_core(
        predicted_probability,
        true_involvement, predicted_involvement=None,
        metric_list=('auc', 'f1', 'mcc', 'sen', 'spe', 'acc_b'),
        current_epoch=None, verbose=False, scores=None, declare_thr=.5, is_involvement=True,
        set_name=None) -> dict:
    predicted_involvement = predicted_probability if predicted_involvement is None else predicted_involvement
    core_predictions = np.array([item > declare_thr for item in predicted_probability])
    core_labels = np.array([item > 0 for item in true_involvement])

    cm = confusion_matrix(core_labels, core_predictions)
    cfs_mtx = cm_score(cm)  # tn, fp, fn, tp
    metrics = get_metrics(cfs_mtx)

    scores = {} if scores is None else scores
    for metric in metric_list:
        if metric.lower() in ['auc', 'auprc']:
            scores[metric] = metrics[metric](core_labels, predicted_probability)
        else:
            scores[metric] = metrics[metric](core_labels, core_predictions)
    scores['corr'] = np.corrcoef(predicted_involvement, true_involvement)[0, 1]
    scores['mae'] = (np.abs(predicted_involvement - true_involvement)).sum()

    if verbose:
        df = pd.DataFrame([scores.values()], columns=[_.upper() for _ in scores.keys()],
                          index=[f'Set = {set_name if set_name else "Best"}'])
        print(cm)
        print(df.round(3))
    return scores


def compute_metrics_patches(prediction, ground_truth,
                            metric_list=('auc', 'auprc', 'f1', 'mcc', 'sen', 'spe', 'pre', 'acc', 'acc_b'),
                            current_epoch=None, verbose=False, scores=None, set_name=None) -> dict:
    prediction_binary = prediction > .5
    cm = confusion_matrix(ground_truth, prediction_binary)
    cfs_mtx = cm_score(cm)  # tn, fp, fn, tp
    # cfs_mtx = None
    # if set_name.lower() == 'test':
    print('\n')
    print(cm)
    metrics = get_metrics(cfs_mtx)
    # metric_list = ('acc_b', 'mcc')
    scores = {} if scores is None else scores
    for metric in metric_list:
        if metric in ['auc', 'auprc']:
            scores[metric] = metrics[metric](ground_truth, prediction)
        else:
            scores[metric] = metrics[metric](ground_truth, prediction_binary)
    if verbose:
        df = pd.DataFrame([scores.values()], columns=[_.upper() for _ in scores.keys()],
                          index=[f'Epoch = {current_epoch if current_epoch else "Best"}'])
        print(df.round(3))
    return scores


if __name__ == '__main__':
    y_true = [0, 1, 1, 1, 0]
    y_pred = [1, 0, 1, 1, 0]
    target_names = ['benign', 'cancer']
    print(confusion_matrix(y_true, y_pred))
