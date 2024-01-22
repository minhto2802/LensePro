import os
import pickle
from typing import Union
from patchify import patchify, unpatchify

import h5py
import numpy as np
import pylab as plt
from skimage import measure
from scipy.signal import hilbert
from numpy.ma import masked_array
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, diamond

from .bm_rf_convert import RFtoBM, BMtoRF
from .misc import timer, crop_bbox, paste_bbox


def load_cores(pid=2, pth=r'/media/minh/My Passport1/workspace/TeUS/ProstateVGH-2/Datanumpy_files',
               verbose=False, core_id=None):
    """Load all available cores of a patient from the storage"""
    _cores = []
    pth = '/'.join((pth, f'Patient{pid:03d}'))

    core_id_list = [core_id, ] if core_id else list(range(0, 15))
    for core_id in core_id_list:
        if os.path.isfile(f'{pth}/core{core_id:02d}.pkl'):
            with open(f'{pth}/core{core_id:02d}.pkl', 'rb') as fp:
                core = pickle.load(fp)

                if verbose:
                    print('Core ID: ', core.metadata['id'], ', Shape: ', core.rf.shape, core.bm.shape)

                core_ = Core()
                core_.update(core)
            _cores.append(core_)
    return _cores


def gen_core_group_idx():
    """Generate index per 5x5 patches over core of size 100 x 20"""
    shape = 100, 20
    roi_b_mask = np.ones(shape)
    patches = patchify(roi_b_mask, (5, 5), 5)
    group_idx = np.arange(np.prod(patches.shape[:2])).reshape(((patches.shape[:2]) + (1, 1)))
    patches *= group_idx
    roi_b_mask = unpatchify(patches, shape)
    return roi_b_mask.flatten()


def get_largest_cc(segmentation):
    """Keep only the largest connected component"""
    labels = measure.label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def gen_mask_index(core):
    """
    Generate the mask index (from 1.. N, N = ROI-height x ROI-width) in B-mode space, then converted to RF space
    The size of mask index is the same with ROI and WP
    :param core: a Core object
    :return: index mask in RF space
    """
    original_shape = core.metadata['original_shape']
    shape = [1, ] + list(original_shape[1:])
    roi = paste_bbox(getattr(core, 'roi'), shape, core.metadata['bbox'])

    prb_radius, arc, rf_depth, resolution = \
        [core.metadata[k] for k in ['prb_radius', 'arc', 'rf_depth', 'resolution']]
    roi_b = RFtoBM(roi.squeeze(), prb_radius, arc, rf_depth, resolution, method='nearest').astype('long')

    mask_b = np.zeros_like(roi_b)
    group_idx = gen_core_group_idx()
    mask_b[roi_b == 1] = group_idx[:roi_b.sum()]
    mask = crop_bbox(BMtoRF(mask_b, prb_radius, arc, rf_depth, resolution, 'nearest'), core.metadata['bbox'])

    return mask


@timer
def load_cores_h5py(patient_id: int, pth: str, core_indices: list = None, suffix=''):
    """Load data from H5PY. This is sufficient to serve training purposes, not visualization.
    Pros: Fast & Convenient (pre-cropped RF images)
    Cons: current files do not contain B-Mode, core location, nor original RF images
    """

    pth = '/'.join((pth, 'h5py' + suffix, f'Patient{patient_id:03d}'))
    print(pth)
    cores = []
    core_indices = range(12) if core_indices is None else core_indices
    for i in core_indices:  # enumerate(cores):
        filename = f'{pth}/core{i}.h5'
        if not os.path.isfile(filename):
            continue
        with h5py.File(filename, 'r') as hf:
            rf = hf['rf'][:]
            wp = hf['wp'][:]
            roi = hf['roi'][:]
            metadata = eval(hf['metadata'][()])  # .decode()
        try:
            core = Core(rf=rf, roi=roi, wp=wp, metadata=metadata,
                        patient_id=patient_id, core_id=i)
            cores.append(core)
        except:
            print(f'Error in loading data. Skip core {i}.')
    return cores


def rf2bm_wrapper_decorator(func):
    """
    This function first checks if RF images (and others) being converted are in original shape.
    If not, images will first be pasted in to images of the original shape, then cropped back their current shape
    """

    def paste_and_crop(core, heatmap: np.ndarray = None, **kwargs):
        original_shape = core.metadata['original_shape'] if 'original_shape' in core.metadata.keys() else core.rf.shape

        if core.rf.shape == original_shape:
            return func(core, heatmap, **kwargs)

        # Paste
        img_list = ['rf', 'wp', 'roi', 'heatmap', 'rf_b', 'wp_b', 'roi_b', 'heatmap_b']
        try:
            core.heatmap = heatmap if heatmap is not None else core.heatmap
        except:
            [img_list.remove(s) for s in ['heatmap', 'heatmap_b']]
            core.heatmap = None

        for attr in img_list[:len(img_list) // 2]:
            shape = original_shape if attr == 'rf' else [1, ] + list(original_shape[1:])
            setattr(core, attr, paste_bbox(getattr(core, attr), shape, core.metadata['bbox']))

        # Registration
        core = func(core, core.heatmap, representative_rf=core.rf[-1], **kwargs)

        # Crop
        bbox_b = regionprops(core.wp_b)[0].bbox
        for attr in img_list:
            if getattr(core, attr) is not None:
                bbox = bbox_b if '_b' in attr else core.metadata['bbox']
                setattr(core, attr, crop_bbox(getattr(core, attr), bbox))

        return core

    return paste_and_crop


@rf2bm_wrapper_decorator
def rf2bm_wrapper(core, heatmap: np.ndarray = None, verbose: bool = False, force: bool = False,
                  representative_rf: np.ndarray = None, quick_convert=False):
    """
    Wrapper of RFtoBM to register the prostate mask, ROI, and RF to B-Mode coordinate
    :param core:
    :param heatmap:
    :param verbose:
    :param force:
    :param representative_rf:
    :param quick_convert: always use 'nearest' for interpolation
    :return:
    """

    if verbose:
        print('Register data to B-Mode coordinate...')

    representative_rf = core.rf[..., -1] if representative_rf is None else representative_rf
    prb_radius, arc, rf_depth, resolution = \
        [core.metadata[k] for k in ['prb_radius', 'arc', 'rf_depth', 'resolution']]

    if (core.rf_b is not None) and not force:
        if verbose:
            print('RF was registered to B-Mode coordinate...')
    else:
        # RF
        analytical_signal = np.abs(hilbert(np.squeeze(representative_rf))) ** 0.3
        core.rf_b = np.asarray(RFtoBM(analytical_signal, prb_radius, arc, rf_depth, resolution,
                                      method='linear' if not quick_convert else 'nearest',
                                      ),
                               dtype=np.float32)
        core.rf_b = np.nan_to_num(core.rf_b)

        # Prostate mask
        wp = get_largest_cc(core.wp.squeeze())
        wp = np.pad(wp, 1)
        core.wp_b = np.asarray(
            RFtoBM(wp, prb_radius, arc, rf_depth, resolution, method='nearest'),
            dtype=np.uint8)
        # ROI
        core.roi_b = np.asarray(RFtoBM(core.roi.squeeze(), prb_radius, arc, rf_depth, resolution, method='nearest'),
                                dtype=np.uint8)
        core.roi_b *= core.wp_b

    # Heat map (2D grayscale)
    if heatmap is not None:
        core.heatmap_b = np.asarray(RFtoBM(heatmap.squeeze(), prb_radius, arc, rf_depth, resolution, method='linear'))

    if verbose:
        print('Done')
    return core


def show_core(core, fig_num=0, scale_hw=3e-2, scale_min=1.0, scale_max=1.0, title: str = None,
              heatmap_name='', heatmap_range=(None, None)):
    """Display prostate mask and ROI on the RF and B-Mode images (if available)
    param:: core: must have 'rf_b', 'wp_b', and 'roi_b' fields; 'bm' field is optional
    """
    h, w = core.bm.shape[:2]
    md = core.metadata

    image_list = [(core.rf_b, 'RF'), ]
    if hasattr(core, 'bm'):
        #         img_bm = core.bm[..., 10] if not md['Revert'] else np.fliplr(core.bm[..., 10])
        img_bm = np.fliplr(core.bm[..., 10])
        image_list.append([img_bm, 'B-Mode'])

    if hasattr(core, 'heatmap'):
        image_list.append([img_bm if hasattr(core, 'bm') else core.rf_b, heatmap_name, heatmap_range])

    fig, ax = plt.subplots(1, len(image_list), num=fig_num, figsize=(scale_hw * h, scale_hw * w * 2))

    for idx, img in enumerate(image_list):
        vmin, vmax = img[0].min() * scale_min, img[0].max() * scale_max

        title = ' | '.join((f'CoreID: {md["id"]}', img[1],
                            'Benign' if md['TrueLabel'] == 0 else
                            f'Cancer [GS={md["GleasonScore"]}, Inv={md["Involvement"]}]'))

        ax[idx].imshow(np.flipud(img[0]) if idx == 0 else img[0], cmap='gray', vmin=vmin, vmax=vmax)
        ax[idx].contour(np.flipud(core.wp_b), colors='b', linewidths=.2)
        ax[idx].contour(np.flipud(core.roi_b), colors='r', linewidths=.2)
        ax[idx].set_title(title, fontsize=22)
        ax[idx].axis('off')

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    return fig, ax


def validate_binary_mask(arr: np.ndarray) -> np.ndarray:
    """
    Remove + Binarize the input array if necessary
    :param arr:
    :return: binarized mask
    """
    arr = np.nan_to_num(arr) if np.any(np.isnan(arr)) else arr
    if (len(np.unique(arr)) > 2) or (arr.max() > 1):
        # Sometimes the intensity values are not constant
        arr = np.array(((arr == 1) + (arr > 128)) > 0, dtype='uint8')
    return arr


def show_image(img: np.ndarray, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)


def center_paste(img, new_shape=(390, 550)):
    h1, w1 = new_shape
    h, w = img.shape
    h1 = max(h1, h+10)
    w1 = max(w1, w+10)
    return np.pad(img,
                  (((h1 - h) // 2, h1 - (h + (h1 - h) // 2)),
                   ((w1 - w) // 2, w1 - (w + (w1 - w) // 2)))
                  )


def review_cores_v0(cores, n_cols=3, fig_num=1, in_b_mode=True, dpi=100):
    """
    Showing multiple cores of a patient
    :param cores: a list of Core object
    :param n_cols: number of columns
    :param fig_num: index of the figure
    :param in_b_mode: whether to display registered (to b-mode coordinates) or raw images
    :param dpi:
    :param figure_dir:
    :param patient_id:
    :return:
    """
    n_cols = int(min(len(cores), n_cols))
    n_rows = int(np.ceil(len(cores) / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, num=fig_num, dpi=dpi)
    try:
        ax = ax.flatten()
    except:
        ax = [ax, ]
    suffix = '_b' if in_b_mode else ''

    for i, c in enumerate(cores):
        if in_b_mode:
            c.rf_b *= c.wp_b
            for k in ['rf', 'wp', 'roi', 'heatmap']:
                setattr(c, f'{k}_b', center_paste(getattr(c, f'{k}_b')))

        ax[i].imshow(np.flipud(eval(f'c.rf{suffix}')), cmap='gray')
        # ax[i].contour(np.flipud(eval(f'c.wp{suffix}')), colors='yellow', linewidths=.3)
        ax[i].contour(np.flipud(eval(f'c.roi{suffix}')), colors='red', linewidths=.3)

        heatmap = masked_array(eval(f'c.heatmap{suffix}'), eval(f'c.roi{suffix}') == 0)
        predicted_inv = (eval(f'c.heatmap{suffix}') > .5).sum() / (eval(f'c.roi{suffix}') == 1).sum()
        ax[i].imshow(np.flipud(heatmap), cmap='jet', vmin=0, vmax=1)

        ax[i].set_title(
            f'Core {c.core_id} | {"Cancer" if c.label else "Benign"} | GS {c.metadata["PrimarySecondary"]} | {predicted_inv:.2f} [{c.metadata["CalculatedInvolvement"]}]',
            fontsize=8)
    [_.axis('off') for _ in ax]
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.tight_layout()
    # plt.suptitle(f'Patient {cores[0].patient_id}')
    plt.show()
    return fig


def review_cores(cores, in_b_mode=True, dpi=250, figure_dir='/', patient_id=0):
    """
    Showing multiple cores of a patient
    :param cores: a list of Core object
    :param in_b_mode: whether to display registered (to b-mode coordinates) or raw images
    :param dpi:
    :param figure_dir:
    :param patient_id:
    :return:
    """
    new_shape = (390, 550)
    suffix = '_b' if in_b_mode else ''

    for i, c in enumerate(cores):
        if in_b_mode:
            c.rf_b *= c.wp_b
            for k in ['rf', 'wp', 'roi', 'heatmap']:
                setattr(c, f'{k}_b',
                        center_paste(getattr(c, f'{k}_b'), new_shape))
        fig = plt.figure(i, dpi=dpi, frameon=False,
                         # figsize=(12 * (new_shape[1]/new_shape[0]), 12)
                         )
        ax = plt.gca()
        plt.imshow(np.flipud(eval(f'c.rf{suffix}')), cmap='gray')
        plt.contour(np.flipud(eval(f'c.roi{suffix}')), colors='red', linewidths=.3)
        heatmap = masked_array(eval(f'c.heatmap{suffix}'), eval(f'c.roi{suffix}') == 0)
        predicted_inv = (eval(f'c.heatmap{suffix}') > .5).sum() / (eval(f'c.roi{suffix}') == 1).sum()
        plt.imshow(np.flipud(heatmap), cmap='jet', vmin=0, vmax=1)

        ax.axis('off')
        plt.text(10, 70,
                 f'True Inv: {c.metadata["CalculatedInvolvement"]:.2f}\n'
                 f'Predicted Inv: {predicted_inv:.2f}',
                 fontsize=28, fontweight='normal', color='yellow')
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        figure_filename = '/'.join(
            (figure_dir, f'Patient{patient_id}_c{c.core_id}_{"Cancer" if c.label else "Benign"}_'
                         f'GS{c.metadata["PrimarySecondary"]}.png'))
        fig.savefig(figure_filename, bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)


def typed_property(name, expected_type, is_bin=False):
    storage_name = '_' + name
    is_bin = is_bin

    @property
    def prop(self):
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value):
        if value is None:
            setattr(self, storage_name, value)
        else:
            if not isinstance(value, expected_type):
                raise TypeError(f'{name} must be a {expected_type}')
            else:
                setattr(self, storage_name, validate_binary_mask(value) if is_bin else value)

    return prop


class Core:
    rf = typed_property('rf', np.ndarray)
    wp = typed_property('wp', np.ndarray, is_bin=True)
    roi = typed_property('roi', np.ndarray, is_bin=True)
    rf_b = typed_property('rf_b', np.ndarray)
    wp_b = typed_property('wp_b', np.ndarray, is_bin=True)
    roi_b = typed_property('roi_b', np.ndarray, is_bin=True)
    metadata = typed_property('metadata', dict)

    def __init__(self, rf: np.ndarray = None, bm: np.ndarray = None,
                 label: int = None, loc: str = None,
                 roi: np.ndarray = None, wp: np.ndarray = None,
                 inv: float = None, gs: str = None, metadata: dict = {},
                 rf_b: np.ndarray = None,
                 wp_b: np.ndarray = None,
                 roi_b: np.ndarray = None,
                 patient_id: int = None,
                 core_id: int = None):
        """

        :param rf: RF in RF plane
        :param bm: B-mode in b-mode plane
        :param metadata:
        :param label:
        :param inv:
        :param loc:
        :param roi:
        :param wp:
        :param gs:
        :param patient_id:
        :param core_id:
        :param rf_b: RF in the B-mode plane
        :param wp_b: prostate mask in the B-mode plane
        :param roi_b: ROI in the B-mode plane
        """
        self.rf = rf
        self.bm = bm
        self.metadata = metadata
        self.loc = loc
        self.roi = roi
        self.wp = wp
        self.inv = inv
        self.label = label
        self.gs = gs
        self.rf_b = rf_b
        self.wp_b = wp_b
        self.roi_b = roi_b
        self.patient_id = patient_id
        self.core_id = core_id

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: Union[str, int]):
        if isinstance(label, str):
            self._label = 1 if label == 'cancer' else 0
        elif isinstance(label, float) or isinstance(label, int):
            self._label = 1 if label > 0 else 0
        elif (label is None) and isinstance(self.inv, float):
            self._label = int(self.inv > .0)
        else:
            self._label = label

    @property
    def gs(self):
        return self._gs

    @gs.setter
    def gs(self, gs: Union[str, int]):
        if isinstance(gs, str):
            self._gs = gs
        elif (gs is None) and ('GleasonScore' in self.metadata.keys()):
            self._gs = self.metadata['GleasonScore']
        else:
            self._gs = gs

    @property
    def inv(self):
        return self._inv

    @inv.setter
    def inv(self, inv: Union[str, int]):
        if isinstance(inv, float):
            try:
                assert 0 <= inv <= 1.
            except:
                print(inv)
                exit()
            self._inv = inv
        elif (inv is None) and ('Involvement' in self.metadata.keys()):
            self._inv = float(self.metadata['Involvement'])
        else:
            self._inv = inv

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc: Union[str, int]):
        if isinstance(loc, str):
            self._loc = loc
        elif (loc is None) and ('Location' in self.metadata.keys()):
            self._loc = self.metadata['gs']
        else:
            self._loc = loc

    @property
    def rf_signal(self):
        if self.roi is None:
            print('ROI is not available. Cannot extract the RF signal.')
        return self.rf[(self.roi * self.wp) > 0]

    def show_roi(self):
        show_image(self._roi.squeeze())
        plt.show()

    def show_wp(self):
        plt.imshow(self._wp.squeeze(), cmap='gray')
        show_image(self._wp.squeeze())
        plt.show()

    def update(self, core: object):
        self.__dict__.update(core.__dict__)


def main():
    c = Core(rf=np.array([3, 4, 5]))
    print(c.rf)


if __name__ == '__main__':
    main()
