import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['plot_train_val', 'plot_confusion_matrix', 'net_interpretation', 'uncertainty_plot']

import numpy as np
import pylab as plt
from sklearn.metrics import confusion_matrix


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_train_val(loss_trn, acc_trn, acc_val, result_dir=None):
    # plot validation metrics
    f, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(loss_trn, label='Train loss')
    ax[0].set_title('Train Loss History')
    ax[0].set_xlabel('Epoch no.')
    ax[0].set_ylabel('Loss')

    ax[1].plot(smooth(acc_trn, 5)[:-2], label='Train acc')
    ax[1].plot(smooth(acc_val, 5)[:-2], label='Val acc')
    ax[1].legend(loc='lower right')
    ax[1].set_title('Accuracy History')
    ax[1].set_xlabel('Epoch no.')
    ax[1].set_ylabel('Accuracy')

    if result_dir is None:
        plt.show()
    else:
        f.savefig(f'{result_dir}/results_CoreN.jpg')


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.get_cmap('Blues'),
                          verbose=False,
                          print_cm_only=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if print_cm_only:
        print(cm)
        return
    #
    #    labels=unique_labels(y_true, y_pred)
    #    labels=labels.astype('int')
    #    # Only use the labels that appear in the data
    #    classes = [classes[i] for i in labels]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose:
            print("Normalized confusion matrix")
    else:
        if verbose:
            print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def net_interpretation(predicted_label, patient_id, involvement, gleason_score, result_dir=None,
                       ood=None, latents=None, declare_thr=.5,
                       cct=(0.2, 0.6, 1), cbt=(0, 1, 0.6), cf=(1, 0.2, 0.6),
                       current_epoch=None, set_name='Test', writer=None, scores: dict = None, to_close=True,
                       core_id=None, core_list=None, mpl_backend=None,
                       ):
    import matplotlib
    if mpl_backend:
        matplotlib.use(mpl_backend)
    else:
        matplotlib.use('Agg')

    if set_name.lower() == 'train':
        if (core_list is not None) and (core_list != 'none'):
            core_list = np.loadtxt(core_list) if isinstance(core_list, str) else core_id
    else:
        core_list = core_id

    # predicted_label = np.array([item > 0.5 for item in predicted_label])
    true_label = np.array([item > 0 for item in involvement])

    current_epoch_str = '' if current_epoch is None else f'_{current_epoch}'
    # auc = roc_auc_score(np.array(true_label), PredictedLabel)
    # print("AUC: %f", auc)
    predicted_label_th = np.array(predicted_label)
    predicted_label_th[predicted_label_th > declare_thr] = 1
    predicted_label_th[predicted_label_th <= declare_thr] = 0
    # plot_confusion_matrix(true_label, predicted_label_th, classes=['Benign', 'Cancer'], title='Confusion matrix',
    #                       print_cm_only=True)
    # plt.savefig(f'{result_dir}/{set_name}_confustion_matrix{current_epoch_str}.png')
    # plt.close()

    andlabels = np.logical_and(predicted_label_th, true_label)
    # norLabels = len(np.where(predicted_label_th + true_label == 0)[0])
    # Acc = (np.sum(andlabels) + norLabels) / len(true_label)
    # Sen = np.sum(andlabels) / np.sum(true_label)
    # Spe = norLabels / (len(true_label) - np.sum(true_label))
    # print("Accuracy: %f" % Acc)
    # print("Sensivity: %f" % Sen)
    # print("Specificity: %f" % Spe)

    patients = np.unique(patient_id)
    Invs = involvement * 100
    gs = np.array(gleason_score)
    indx = []
    maxc = 0
    for ip in patients:
        temp = np.where(patient_id == ip)[0]
        indx.append(temp)
        maxc = max(maxc, len(temp))

    inv = np.zeros((len(patients), maxc), dtype=float)
    #    cmaps=[]

    label = []
    cmap = [cf if True else [0, 0, 1] for i in range(len(true_label))]

    for i in range(len(true_label)):
        if andlabels[i] == 1:
            cmap[i] = cct
        elif (predicted_label_th[i] + true_label[i]) == 0:
            cmap[i] = cbt
        else:
            cmap[i] = cf
    cmap = np.array(cmap)
    cmaps = np.zeros((len(patients), maxc, 3), dtype=float)
    for ip in range(len(patients)):
        indxip = indx[ip]
        inv[ip, :len(indxip)] = Invs[indxip]
        cmaps[ip, :len(indxip)] = cmap[indxip]
        label.append(gs[indxip])
        for i in range(len(label[ip])):
            if label[ip][i] == '-':
                inv[ip, i] = 50
            if label[ip][i] == 'FB':
                inv[ip, i] = 50
                label[ip][i] = '-'

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(18.5 / 2, 10.5 / 2)
    barbase = np.cumsum(np.concatenate((np.zeros((inv.shape[0], 1)), inv[:, 0:-1]), axis=1), 1)

    for i in range(maxc):
        ax1.bar(np.arange(len(patients)), inv[:, i].tolist(), 0.7, bottom=barbase[:, i], color=cmaps[:, i])
        ax1.EdgeColor = 'k'
    plt.xticks(np.arange(len(patients)), patients)
    plt.xlabel('Patient No.')
    # fig1.tight_layout()

    joblblpos = inv / 2 + barbase
    for k1 in range(inv.shape[0]):
        for k2 in range(inv.shape[1]):
            plt.text(k1, joblblpos[k1, k2], label[k1][k2] if inv[k1, k2] != 0 else '')
    # plt.savefig(f'{result_dir}/{set_name}_acc_per_core{current_epoch_str}.png')

    if ood is not None:
        ood_sum = np.array([-_ood.sum() for _ood in ood])
        ood_normalized = ood_sum / ood_sum.sum()
        size = ood_normalized
    else:
        size = None

    fig2 = plt.figure(2)
    ax2 = sns.scatterplot(x=involvement, y=predicted_label, size=size, legend=False, hue=np.isin(core_id, core_list))
    diag = np.arange(0, 1, .05)
    sns.lineplot(x=diag, y=diag, color='r', ax=ax2)
    ax2.axvspan(-.1, 0.1, -.1, declare_thr, alpha=.2, facecolor='lightgreen')
    ax2.axvspan(-.1, 0.1, declare_thr + .01, 1., alpha=.2, facecolor='red')
    ax2.axvspan(0.11, 1.1, -.1, declare_thr, alpha=.2, facecolor='grey')
    ax2.axvspan(0.11, 1.1, declare_thr + .01, 1., alpha=.2, facecolor='moccasin')
    ax2.axvline(x=.101, linewidth=.6, linestyle='--', color='black')
    ax2.axhline(y=declare_thr + .001, linewidth=.6, linestyle='--', color='black')
    ax2.axis('square')
    ax2.set(ylim=[-.1, 1.1], xlim=[-.1, 1.1])
    if scores is not None:
        ax1.set_title(f'ACC_B: {scores["acc_b"]:.2f} '
                      f'AUC: {scores["auc"]:.2f} | SEN: {scores["sen"]:.2f} | SPE: {scores["spe"]:.2f}')
        ax2.set(title=f'Correlation Coefficient = {scores["corr"]:.3f} | MAE = {scores["mae"]:.3f}',
                xlabel='True Involvement', ylabel='Predicted Involvement'
                )
    # fig2.tight_layout()

    # Latent space
    # cmap = ListedColormap(sns.color_palette("tab10")[:2])
    # true_label_rep = np.concatenate([np.repeat(label, feat.shape[0]) for label, feat in zip(true_label, latents)])
    # latents = torch.pca_lowrank(torch.tensor(np.concatenate(latents)), 2)[0]
    # pca = decomposition.PCA()
    # pca.n_components = 2
    # latents = pca.fit_transform(np.concatenate(latents))
    # ood = np.concatenate(ood)
    # ood = (ood - ood.min()) / (ood.max() - ood.min())
    # fig3 = plt.figure(3)
    # ax3 = sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=true_label_rep, size=ood, alpha=.5, cmap=cmap)
    # hl = [(plt.Line2D([], [], color=cmap(i), ls="", marker="o"), cl) for i, cl in enumerate(['Benign', 'Cancer'])]
    # plt.legend(*zip(*hl), title="Class")

    if writer:
        # img = plot_to_image(fig4)
        writer.add_figure(f'{set_name}/core_acc', fig1, global_step=current_epoch)
        writer.add_figure(f'{set_name}/core_inv', fig2, global_step=current_epoch)
        # writer.add_figure(f'{set_name}/latent', fig3, global_step=current_epoch)
    if to_close:
        plt.close('all')


def uncertainty_plot(m_inv, s_inv, PatientId, Inv, GS, CCT=[0.2, 0.6, 1], CBT=[0, 1, 0.6], CF=[1, 0.2, 0.6]):
    patients = np.unique(PatientId)
    gs = np.array(GS)
    indx = []
    maxc = 0
    for ip in patients:
        temp = []
        temp = np.where(PatientId == ip)[0]
        indx.append(temp)
        m_inv_p = m_inv[temp]
        s_inv_p = s_inv[temp]
        Inv_p = Inv[temp]
        plt.title('Patient' + str(ip))
        # plt.figure()
        plt.errorbar(range(len(temp)), m_inv_p, yerr=s_inv_p)
        plt.show()
        plt.xticks(range(len(Inv_p)), Inv_p)
        maxc = max(maxc, len(temp))
