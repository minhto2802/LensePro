import matplotlib
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


class Analyzer:
    def __init__(self, dataset):
        self.dataset = dataset

    def update(self, index, epoch, forget_rate, writer=None):
        if isinstance(index, list):
            index = np.concatenate(index)
        df = self.build_df(index)
        fig1 = plot_non_update_count(df, epoch, forget_rate)
        if writer:
            writer.add_figure(f'Non-update/count', fig1, global_step=epoch)
        plt.close('all')

    def build_df(self, index):
        d = self.dataset
        df = pd.DataFrame([index, d.patient_id[index], d.core_id[index],
                           d.inv[index].numpy(), d.inv[index].numpy() > 0, d.data[index]]).T
        df.columns = ['id', 'pid', 'cid', 'inv', 'label', 'ts']
        df.label.replace({False: 'Benign', True: 'Cancer'}, inplace=True)
        return df


def plot_non_update_count(df, epoch, forget_rate, context='talk', font_scale=1.1):
    # matplotlib.use('TkAgg')
    # plt.close('all')
    fig = plt.figure(num='Non-update count', figsize=(20, 8))
    with sns.plotting_context(context, font_scale=font_scale) as _:
        # Per patient
        ax = sns.countplot(x='pid', hue='label', data=df, hue_order=['Benign', 'Cancer'],
                           palette=sns.color_palette("tab10", 2))
        leg = plt.legend(loc='upper right', title="Class", frameon=False)
        ax.add_artist(leg)
        # Per core
        colors = plt.cm.get_cmap('tab20b').colors[:12]
        sns.countplot(x='pid', hue='cid', data=df, palette=colors)
        h = [mpatches.Patch(color=c) for c in colors]
        plt.legend(handles=h, labels=range(12), loc='upper left', title="Core ID", frameon=False)
        ax.add_artist(leg)

        # Set captions
        n_signals = len(df.index[df.inv > 0].unique())
        ax.set_xlabel('Patient ID')
        ax.set_yticks(np.arange(0, ax.get_ylim()[1], ax.get_ylim()[1]//10))
        ax.set_title(f'Epoch {epoch} | Forget-rate: {forget_rate:.3f} | '
                     f'Num cancer-signals: {n_signals} ({n_signals/len(df.index.unique()):.2f}%) '
                     f'({len(df.pid[df.inv > 0].unique())} patients)')
        plt.tight_layout()

    # plt.show()
    return fig
