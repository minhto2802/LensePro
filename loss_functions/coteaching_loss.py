import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from loss_functions.isomaxplus import IsoMaxPlusLossSecondPart as IsoMaxLossSecondPart
# from loss_functions.isomax import IsoMaxLossSecondPart

from utils.get_loss_function import get_loss_function
from utils.dataloader import make_weights_for_balanced_classes


# criteria = torch.nn.CrossEntropyLoss(reduction='none')
# criteria = get_loss_function('nce_agce')
# criteria = get_loss_function('agce')


# Loss function for Co-Teaching
def loss_coteaching(
        y_1,
        y_2,
        t,
        forget_rate,
        class_weights=None,
        num_classes=2,
        get_index_only=False,
        weigh_loss=False,
        mask=None,
        **kwargs
):
    """Co-Teaching Loss function.

    Parameters
    ----------
    y_1 : Tensor array
      Output logits from model 1

    y_2 : Tensor array
      Output logits from model 2

    t : np.array
      List of Noisy Labels (t means targets)

    forget_rate : float
      Decimal between 0 and 1 for how quickly the models forget what they learn.
      Just use rate_schedule[epoch] for this value

    class_weights : Tensor array, shape (Number of classes x 1), Default: None
      A np.torch.tensor list of length number of classes with weights

    num_classes:
    get_index_only:
    """
    extra_outputs = None
    num_remember = len(t)

    # masked_idx = torch.where()
    # mask = torch.nan_to_num(torch.inf * torch.tensor(mask).long(), 1)
    # mask = [torch.nan_to_num(torch.inf * torch.tensor(_mask).long(), 1) for _mask in mask]
    mask = [torch.abs(1-torch.tensor(_mask).long()) for _mask in mask]

    t1, t2 = t.clone(), t.clone()
    # t1[mask[0] == 1] -= 0.5
    # t2[mask[1] == 1] -= 0.5
    # t1, t2 = torch.abs(t1), torch.abs(t2)

    loss_func = kwargs['loss_func'] if 'loss_func' in kwargs.keys() else [get_loss_function('ce'),
                                                                          get_loss_function('ce')]
    loss_func1, loss_func2 = loss_func
    if ('ind_1_update' not in kwargs.keys()) or ('ind_2_update' not in kwargs.keys()):
        # loss_1 = loss_func1(y_1, t, reduction='none', class_weights=class_weights)
        # loss_2 = loss_func2(y_2, t, reduction='none', class_weights=class_weights)
        # loss_1 = criteria(y_1, t)
        # loss_2 = criteria(y_2, t)

        # loss_1 = IsoMaxLossSecondPart()(y_1.detach(), t1.detach(), reduction='none')
        # loss_2 = IsoMaxLossSecondPart()(y_2.detach(), t2.detach(), reduction='none')
        loss_1 = -(y_1[range(t1.size(0)), t1.argmax(1)] - y_1[range(t1.size(0)), t1.argmin(1)])
        loss_2 = -(y_2[range(t2.size(0)), t2.argmax(1)] - y_2[range(t2.size(0)), t2.argmin(1)])

        loss_1 *= mask[1].cuda(loss_1.device)
        loss_2 *= mask[0].cuda(loss_2.device)

        ind_1_sorted = np.argsort(loss_1.data.cpu())
        ind_2_sorted = np.argsort(loss_2.data.cpu())
        if ('loss_thr' in kwargs) and (np.prod(kwargs['loss_thr']) < np.Inf):
            ind_1_update, ind_2_update = [], []
            for c in range(num_classes):
                ind_1_update.append(np.argwhere(loss_1.data.cpu() <= kwargs['loss_thr'][0][c])[0])
                ind_2_update.append(np.argwhere(loss_2.data.cpu() <= kwargs['loss_thr'][1][c])[0])
            ind_1_update, ind_2_update = torch.cat(ind_1_update), torch.cat(ind_2_update)
        else:
            loss_1_sorted = loss_1[ind_1_sorted]

            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * len(loss_1_sorted))

            ind_1_update = ind_1_sorted[:num_remember]
            ind_2_update = ind_2_sorted[:num_remember]

            # ind_12 = torch.unique((torch.cat([ind_1_sorted[num_remeamber:], ind_2_sorted[num_remember:]])))
            # Share updates between the two models.
            # TODO: these class weights should take into account the ind_mask filters.

            # Compute class weights to counter class imbalance in selected samples
            # class_weights_1 = estimate_class_weights(t[ind_1_update], num_class=num_classes)
            # class_weights_2 = estimate_class_weights(t[ind_2_update], num_class=num_classes)

        # Equalizing the number of instances in every class

        # ind_2_update = ind_2_update[mask[0][ind_2_update].bool()]
        # ind_1_update = ind_1_update[mask[1][ind_1_update].bool()]

        if not weigh_loss:
            try:
                ind_1_update = balance_clipping(t1.argmax(1), ind_1_update, num_classes)
                ind_2_update = balance_clipping(t2.argmax(1), ind_2_update, num_classes)
            except:
                pass

        if 'extra_outputs' in kwargs.keys():
            extra_outputs = {'cutoff_thr': [], 'min_loss': [], 'num_kept': [], 'loss': [],
                             'ind_non_update': get_common_non_update_index(ind_1_sorted[num_remember:],
                                                                           ind_2_sorted[num_remember:], percent=1),
                             'ind_common_update': np.intersect1d(ind_1_update, ind_2_update),
                             }  # for two networks
            extra_outputs['perc_common_update'] = \
                len(extra_outputs['ind_common_update']) / len(np.union1d(ind_1_update, ind_2_update))
            # extra_outputs = get_extra_outputs(kwargs['extra_outputs'],
            #                                   loss_1, loss_2, t, ind_1_update, ind_2_update,
            #                                   num_classes)
            # print(extra_outputs['ind_non_update'])

        if get_index_only:
            return ind_1_update, ind_2_update

        # Randomly replacing indices of benign samples
        # ind_1_update = random_replacing(t, ind_1_update)
        # ind_2_update = random_replacing(t, ind_2_update)
    else:
        ind_1_update, ind_2_update = kwargs['ind_1_update'], kwargs['ind_2_update']

    if (len(ind_1_update) * len(ind_2_update)) == 0:
        return (
            # (loss_1 * (loss_1 < kwargs['loss_thr'][0])).mean(),
            # (loss_2 * (loss_2 < kwargs['loss_thr'][0])).mean(),
            0, 0, extra_outputs,
        )

    # loss_1_update = loss_func1(y_1[ind_2_update], t[ind_2_update], reduction='none', class_weights=class_weights)
    # loss_2_update = loss_func2(y_2[ind_1_update], t[ind_1_update], reduction='none', class_weights=class_weights)
    # loss_1_update = criteria(y_1[ind_2_update], t[ind_2_update])
    # loss_2_update = criteria(y_2[ind_1_update], t[ind_1_update])
    loss_1_update = IsoMaxLossSecondPart()(y_1[ind_2_update], t1[ind_2_update],
                                           reduction='none')
    loss_2_update = IsoMaxLossSecondPart()(y_2[ind_1_update], t2[ind_1_update],
                                           reduction='none')

    if weigh_loss:
        loss_1_update *= make_weights_for_balanced_classes(torch.tensor(t1[ind_2_update].argmax(-1))).to('cuda')
        loss_2_update *= make_weights_for_balanced_classes(torch.tensor(t2[ind_1_update].argmax(-1))).to('cuda')

    # loss_1_update = loss_func1(
    #     y_1[ind_2_update], t[ind_2_update], reduction='none', weight=None,  # weight=class_weights_2,
    #     sub_index=ind_2_update, **kwargs)
    # loss_2_update = loss_func2(
    #     y_2[ind_1_update], t[ind_1_update], reduction='none', weight=None,  # weight=class_weights_1,
    #     sub_index=ind_1_update, **kwargs)

    # loss_1_update *= kwargs['loss_weights'][ind_2_update]
    # loss_2_update *= kwargs['loss_weights'][ind_1_update]

    if 'anneal' in kwargs.keys():
        anneal_loss = kwargs['anneal'].anneal_loss
        loss_1_update = anneal_loss(y_1[ind_2_update], t1[ind_2_update], loss_1_update, kwargs['global_step'])
        loss_2_update = anneal_loss(y_2[ind_1_update], t2[ind_1_update], loss_2_update, kwargs['global_step'])

    return (
        # torch.sum(loss_1_update) / len(loss_1_update),
        # torch.sum(loss_2_update) / len(loss_2_update),
        loss_1_update.mean(),
        loss_2_update.mean(),
        # loss_1_update.sum() / mask[1][ind_2_update].sum(),
        # loss_2_update.sum() / mask[0][ind_1_update].sum(),
        extra_outputs,
        # get_chosen_index(t, ind_1_update, ind_2_update),
        # ind_12
    )


def loss_coteaching_plus(logits, logits2, labels, forget_rate, class_weights=None, **kwargs):
    step = kwargs['step']
    loss_func = kwargs['loss_func'] if 'loss_func' in kwargs.keys() else [F.cross_entropy, F.cross_entropy]
    loss_func1, loss_func2 = loss_func

    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_id.append(idx)
            logical_disagree_id[idx] = True

    temp_disagree = logical_disagree_id.astype(np.int64)
    if 'index' in kwargs.keys():
        temp_disagree *= kwargs['index'].cpu().numpy()
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]
        if 'index' in kwargs.keys():
            kwargs['index'] = kwargs['index'][disagree_id]

        loss_1, loss_2, chosen_ind = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate,
                                                     **kwargs)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        loss_1_update = loss_func1(
            update_outputs, update_labels, reduction='none', weight=class_weights, **kwargs)
        loss_2_update = loss_func2(
            update_outputs2, update_labels, reduction='none', weight=class_weights, **kwargs)

        loss_1 = torch.sum(update_step * loss_1_update) / labels.size()[0]
        loss_2 = torch.sum(update_step * loss_2_update) / labels.size()[0]
        chosen_ind = get_chosen_index(labels, np.argsort(loss_1.data.cpu()), np.argsort(loss_2.data.cpu()))

    return loss_1, loss_2, chosen_ind


def get_loss_coteaching(loss_name, num_classes, use_plus=False, relax=False, **kwargs_):
    loss_func = [get_loss_function(loss_name, num_classes, **kwargs_) for _ in range(2)]
    cot_func = loss_coteaching_plus if use_plus else loss_coteaching

    def wrapper(*args, **kwargs):
        return cot_func(*args, **kwargs, loss_func=loss_func, relax=relax, num_classes=num_classes)

    return wrapper


def get_chosen_index(target, ind_1, ind_2):
    ind = {
        'benign': [np.argwhere(target[ind_1].cpu() == 0)[0], np.argwhere(target[ind_2].cpu() == 0)[0]],
        'cancer': [np.argwhere(target[ind_1].cpu() == 1)[0], np.argwhere(target[ind_2].cpu() == 1)[0]],
        'cancer_ratio': [target[ind_1].sum().cpu().item() / len(ind_1),
                         target[ind_2].sum().cpu().item() / len(ind_2)]
    }
    return ind


def random_replacing(label, index, target_class=0):
    """
    Randomly replace the index of the target class
    :param label:
    :param index:
    :param target_class:
    :return:
    """
    target_idx = np.argwhere((label == target_class).cpu())[0]
    length = len(index[label[index] == target_class])
    new_idx = np.random.choice(target_idx, length, replace=False)
    index[label[index] == target_class] = torch.tensor(new_idx)
    return index


def balance_clipping(label, index, num_classes=2):
    """
    :param label:
    :param index:
    :param num_classes:
    :return:
    """
    if len(index) == 0:
        return index

    min_num = torch.histc(label[index], num_classes).min().item()
    index_b = []
    for k in range(num_classes):
        idx_chosen = np.random.permutation(label[index])[:min_num]
        index_b.append(index[label[index] == k][idx_chosen])
        # index_b.append(index[label[index] == k][:min_num])
    if len(torch.cat(index_b)) == 0:  # in case one of the class doesn't have any samples
        # return index
        return []
    return torch.cat(index_b)


def estimate_class_weights(label, num_class: int):
    """

    :param label: 1D array
    :param num_class: int
    :return:
    """
    freq_inv = 1 / (torch.histc(label, num_class) / len(label))
    class_weights = (freq_inv / freq_inv.sum()) / (1 / num_class)
    return class_weights


def get_extra_outputs(extra_outputs, loss_1, loss_2, t, ind_1_update, ind_2_update, num_classes):
    """

    :param extra_outputs:
    :param loss_1:
    :param loss_2:
    :param t:
    :param ind_1_update:
    :param ind_2_update:
    :param num_classes:
    :return:
    """
    min_loss, max_loss, loss = [], [], []
    for c in range(num_classes):
        # if (len(ind_1_update) * len(ind_2_update)) == 0:
        #     loss_1_c = loss_1[t == c].cpu().detach()
        #     loss_2_c = loss_2[t == c].cpu().detach()
        # else:
        #     loss_1_c = loss_1[ind_1_update][t[ind_1_update] == c].cpu().detach()
        #     loss_2_c = loss_2[ind_2_update][t[ind_2_update] == c].cpu().detach()
        # max_loss_1, min_loss_1 = loss_1_c.max().item(), loss_1_c.min().item()
        # max_loss_2, min_loss_2 = loss_2_c.max().item(), loss_2_c.min().item()
        # max_loss.append([max_loss_1, max_loss_2])
        # min_loss.append([min_loss_1, min_loss_2])
        loss.append([loss_1[t == c].cpu().detach(), loss_2[t == c].cpu().detach()])
    # extra_outputs['cutoff_thr'].append(max_loss)
    # extra_outputs['min_loss'].append(min_loss)
    extra_outputs['num_kept'].append([len(ind_1_update) / len(t), len(ind_2_update) / len(t)])
    extra_outputs['loss'].append(loss)
    return extra_outputs


def get_common_non_update_index(ind_1: torch.Tensor, ind_2: torch.Tensor, percent=.1):
    """
    Select "percent" percentage of ind_1 & ind_2, then find the overlapped (non-updated by any network) indices
    :param ind_1:
    :param ind_2:
    :param percent:
    :return:
    """
    n = int(percent * len(ind_1))
    if n == 0:
        return []
    ind_1, ind_2 = ind_1[:n], ind_2[:n]
    return np.intersect1d(ind_1, ind_2)


def get_filtered_idx(batch_size, update_idx):
    """
    Return the list of indices of samples that were filtered by co-teaching
    """
    all_idx = torch.range(0, batch_size - 1)
    filtered_idx = all_idx[~torch.isin(all_idx, update_idx)]
    return filtered_idx
