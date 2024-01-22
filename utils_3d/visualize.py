import numpy as np
from scipy.signal import hilbert

import matplotlib
import pylab as plt

# matplotlib.use('TkAgg')


def make_analytical(x):
    return np.abs(hilbert(x)) ** 0.3


def visualize_patches(x, num_rows: int = 4, fig_size: tuple = (2, 2), ts_idx: int = 199, cmap='gray'):
    """
    visualize an entire biopsy core
    :param x: NP x TS x H x W
    :param num_rows
    :param fig_size: (H, W)
    :param ts_idx
    :param cmap
    :return:
    """
    assert num_rows > 0
    num_cols = int(np.ceil(x.shape[0]) / 4)
    fig = plt.figure(figsize=(fig_size[1] * num_cols, fig_size[0] * num_rows))
    ax = fig.subplots(num_rows, num_cols)
    ax = ax.flatten()
    x_vis = make_analytical(np.squeeze(x[:, ts_idx]))
    d_range = x_vis.min(), x_vis.max()
    for i, (_x_vis, _ax) in enumerate(zip(x_vis, ax)):
        _ax.imshow(_x_vis, cmap=cmap, vmin=d_range[0], vmax=d_range[1])
        _ax.text(1, 3, i, color='cyan')
        # _ax.axis('off')
        _ax.set_xticks([]), _ax.set_yticks([])
        plt.imsave(f'tmp/{i:02d}.png', _x_vis, cmap=cmap, vmin=d_range[0], vmax=d_range[1])
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.show()
    return fig, ax


def visualize_views(views, rf, roi_org, roi):
    """

    :param views: output of utils_3d.dataset.gen_views with return_mask=True
    :param rf: HWT
    :param roi_org: NHW (original ROI)
    :param roi: NHW (new ROI)
    Usage:
        from utils_3d.visualize import visualize_views
        views = []
        for i in range(n_views):
            views.extend(extract_patches(rf, (roi.shape, rows, cols),
                                         patch_info, num_ts, inv, shift_perc, shift_ranges[i], random_scale, grid=grid,
                                         return_mask=True))
        rf, roi_org = load_raw_npy(filename, patch_info)
        visualize_views(views, rf, roi_org, roi)
    """
    import pylab as plt
    x_vis = make_analytical(np.squeeze(rf[..., 199]))
    plt.figure(1)
    plt.imshow(np.flipud(x_vis), cmap='gray')
    plt.contour(np.flipud(roi_org), colors='red')
    plt.contour(np.flipud(roi), colors='blue')

    plt.figure(2)
    plt.imshow(np.flipud(x_vis), cmap='gray')
    for i in range(len(views[0])):
        plt.contour(np.flipud(views[1][i]), colors='orange')
        plt.contour(np.flipud(views[3][i]), colors='cyan')
    plt.show()
