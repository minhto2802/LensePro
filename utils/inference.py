import numpy as np


def infer_core_wise(inputs, predictions, core_len, roi_coors=None, ood_scores=None, features=None):
    """
    Infer core-wise predictions based on signal-wise predictions.
    Input must be 1-D array
    :param inputs: signal-wise inputs
    :param predictions: signal-wise predictions
    :param core_len: length of each core
    :param roi_coors: pixel-wise coordinates of ROI area
    :param ood_scores:
    :param features:
    :return:
    """
    counter = 0
    core_feat1, core_res1, core_l1, core_ood, core_feat2 = [], [], [], [], []
    core_inputs = []

    # find a label for each core
    for i in range(len(core_len)):
        temp = predictions[counter:(counter + core_len[i])]
        core_res1.append(temp)
        core_l1.append(np.greater(temp[:, 1], temp[:, 0]).sum() / core_len[i])
        temp = temp[:, 1]
        th = 0.5
        core_feat1.append([temp[temp > th].mean(), len(temp[temp > th]), temp[temp < th].mean(), len(temp[temp < th])])
        core_ood.append(ood_scores[counter:(counter + core_len[i])])
        core_feat2.append(features[counter:(counter + core_len[i])])
        core_inputs.append(inputs[counter:(counter + core_len[i])])
        counter += core_len[i]

        # heatmap = np.zeros((roi_coors[i][0].max()+1, roi_coors[i][1].max()+1))
        # heatmap[roi_coors[i][0], roi_coors[i][1]] = core_res1[i][:, 1]
        # prediction_maps.append(heatmap)
    return core_inputs, core_l1, core_ood, core_feat2, core_res1
