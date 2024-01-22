import random

from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import imgaug as ia
import imgaug.augmenters as iaa


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


class PatchTransform(object):
    def __init__(self):
        def get_transform():
            def _get_transform():
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            224, interpolation=InterpolationMode.BICUBIC
                        ),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ])
                return transform

            return _get_transform

        self.transform = get_transform()()
        self.transform_prime = get_transform()()

    def __call__(self, sample):
        x1 = self.transform(sample)[np.newaxis]
        x2 = self.transform_prime(sample)[np.newaxis]
        return x1, x2


class OneCropTransform:
    def __init__(self, center_crop=True, size=(32, 32), in_channels=1, n_outputs=1):
        ia.seed(1)
        size = size if isinstance(size, tuple) else (size, size)
        self.in_channels = in_channels
        self.transform = self.get_transform(center_crop, size)
        self.n_outputs = n_outputs

    @staticmethod
    def get_transform(center_crop=True, size=(32, 32)):
        ran_seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips,
            # iaa.Affine(rotate=(-20, 20)),
            # iaa.Affine(translate_px={"x": (-2, 2), "y": (-2, 2)}),
            # iaa.OneOf([
            #     iaa.Rot90([0, 1, 3]),
            #     iaa.Affine(rotate=(-20, 20)),
            # ]),
            iaa.Rot90([0, 1, 3]),
            # iaa.LinearContrast((0.9, 1.1)),
            # iaa.AdditiveGaussianNoise(scale=0.01),
        ], random_order=True)  # apply augmenters in random order
        if center_crop:
            return iaa.Sequential([ran_seq, iaa.CenterCropToFixedSize(*size)])
        return iaa.Sequential([ran_seq, iaa.CropToFixedSize(*size)])

    def __call__(self, sample):
        if (sample.ndim > 2) and (sample.shape[-1] > 1):
            assert sample.shape[-1] >= self.in_channels
            sample = sample[..., :self.in_channels]

        x = self.transform(image=sample)

        if self.n_outputs > 1:
            xs = [x]
            for _ in range(self.n_outputs - 1):
                xs.append(self.transform(image=sample))
            if xs[0].ndim > 2:
                return [_.transpose([2, 0, 1]) for _ in xs]
            return [_[np.newaxis] for _ in xs]
        else:
            if x.ndim > 2:
                return x.transpose([2, 0, 1])
            return x[np.newaxis]


class JustCropTransform(OneCropTransform):
    @staticmethod
    def get_transform(center_crop=True, size=(32, 32)):
        if center_crop:
            return iaa.CenterCropToFixedSize(*size)
        return iaa.CropToFixedSize(*size)


class OneCropTransformSupervise(OneCropTransform):
    def __init__(self, center_crop=True, size=(32, 32), **kwargs):
        super(OneCropTransformSupervise, self).__init__(center_crop, size, **kwargs)

    @staticmethod
    def get_transform(center_crop=True, size=(32, 32)):
        ran_seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips,
            iaa.Rot90([0, 1, 2, 3]),
            # iaa.Crop(percent=(0, 0.1)),  # random crops,
            # iaa.LinearContrast((0.7, 1.3)),
            iaa.LinearContrast((0.6, 1.4)),
            iaa.Affine(
                translate_percent={'x': (-0.2, 0.2)},
                # scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                rotate=(-25, 25),
                #     order=[0, 1],
                #     # shear=(-8, 8),
            ),
            iaa.AdditiveGaussianNoise(scale=0.2 * 1),
            # iaa.Affine(
            #     rotate=(-15, 15),
            # ),
            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 0.1), per_channel=0.5),
            # iaa.Dropout(),
        ], random_order=True)  # apply augmenters in random order
        if center_crop:
            return iaa.Sequential([ran_seq, iaa.CenterCropToFixedSize(*size)])
        return iaa.Sequential([ran_seq,
                               iaa.OneOf([iaa.CenterCropToFixedSize(*size),
                                          iaa.CropToFixedSize(*size)])])


class OneCropTransformV1(OneCropTransform):

    @staticmethod
    def get_transform():
        ran_seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips,
            iaa.Crop(percent=(0, 0.1)),  # random crops,
            # Small gaussian blur with random sigma between 0 and 0.5.,
            # But we only blur about 50% of all images.,
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.,
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.,
            # For 50% of all images, we sample the noise once per pixel.,
            # For the other 50% of all images, we sample the noise per pixel AND,
            # channel. This can change the color (not only brightness) of the,
            # pixels.,
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.,
            # In 20% of all cases, we sample the multiplier once per channel,,
            # which can end up changing the color of the images.,
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.,
            # Scale/zoom them, translate/move them, rotate them and shear them.,
            iaa.Affine(
                scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8),
            ),
        ], random_order=True)  # apply augmenters in random order
        return iaa.Sequential([ran_seq, iaa.CenterCropToFixedSize(32, 32)])
        # return iaa.CenterCropToFixedSize(32, 32)


class OneCropTransformTime(OneCropTransform):
    @staticmethod
    def get_transform(center_crop=True, size=(32, 32)):
        return iaa.Identity()

    def __call__(self, sample):
        assert sample.ndim == 3
        sample = sample.transpose([2, 0, 1])
        idx = np.random.choice(sample.shape[0], size=1, replace=False)
        return self.transform(image=sample[idx[0]])[np.newaxis]


class TwoCropsTransform(OneCropTransform):
    def __init__(self, center_crop=True, crop_size=(32, 32), *args, **kwargs):
        ia.seed(1)
        super(TwoCropsTransform, self).__init__(center_crop, crop_size, *args, **kwargs)
        self.transform_prime = self.get_transform(center_crop, crop_size)
        self.crop_size = crop_size

    def __call__(self, sample):
        if (sample.ndim > 2) and (sample.shape[-1] > 1):
            assert sample.shape[-1] >= self.in_channels
            sample = sample[..., -self.in_channels:]

        x1 = self.transform(image=sample)
        x2 = self.transform_prime(image=sample)

        # visualize_3d_patches(x1, x2, sample)
        if x1.ndim > 2:
            return x1.transpose([2, 0, 1]), x2.transpose([2, 0, 1])
        return x1[np.newaxis], x2[np.newaxis]


class TwoCropsTransformInfoMin(TwoCropsTransform):
    def __init__(self, *args, **kwargs):
        ia.seed(1)
        super(TwoCropsTransformInfoMin, self).__init__(*args, **kwargs)
        self.transform, self.transform_prime = self.info_min_transform(self.crop_size)

    @staticmethod
    def info_min_transform(crop_size):
        aug1 = [
            iaa.Affine(rotate=(10, 25),
                       # translate_percent={'x': (0.1, 0.2), 'y': (-0.2, 0.2)}
                       ),
            iaa.LinearContrast((1.1, 1.5))
        ]
        aug2 = [
            iaa.Affine(rotate=(-25, -10),
                       # translate_percent={'x': (-0.2, -0.1), 'y': (-0.2, 0.2)}
                       ),
            iaa.LinearContrast((0.5, 0.9))
        ]
        aug1 = iaa.Sequential([iaa.Sequential(aug1, random_order=True),
                               iaa.OneOf([
                                   iaa.CropToFixedSize(crop_size[0], crop_size[1], (0.0, 1.0)),
                                   iaa.CropToFixedSize(crop_size[0], crop_size[1], (0.0, 0.5)),
                                   iaa.CropToFixedSize(crop_size[0], crop_size[1], (0.0, 0.0))])
                               ])
        aug2 = iaa.Sequential([iaa.Sequential(aug2, random_order=True),
                               iaa.OneOf([
                                   iaa.CropToFixedSize(crop_size[0], crop_size[1], (1.0, 1.0)),
                                   iaa.CropToFixedSize(crop_size[0], crop_size[1], (1.0, 0.5)),
                                   iaa.CropToFixedSize(crop_size[0], crop_size[1], (1.0, 0.0))])
                               ])
        return aug1, aug2


class FixMatchTransform(TwoCropsTransform):
    def __init__(self, center_crop=True, crop_size=(32, 32), *args, **kwargs):
        ia.seed(1)
        super(FixMatchTransform, self).__init__(*args, **kwargs)
        self.transform = self.get_weak_transform(center_crop, crop_size)
        self.transform_prime = self.get_strong_transform(center_crop, crop_size)

    @staticmethod
    def get_weak_transform(center_crop=True, size=(32, 32)):
        return iaa.Sequential([iaa.Fliplr(0.5), iaa.CenterCropToFixedSize(*size)])

    @staticmethod
    def get_strong_transform(center_crop=True, size=(32, 32)):
        # return iaa.Sequential([iaa.Fliplr(0.5), iaa.CropToFixedSize(*size)])
        # return iaa.Sequential([
        #     iaa.Fliplr(0.5),
        #     # iaa.Rot90([0, 1, 2, 3]),
        #     iaa.SomeOf((1, 3), [
        #         iaa.LinearContrast((0.5, 0.8)),
        #         iaa.LinearContrast((1.2, 1.5)),
        #         iaa.GaussianBlur(sigma=(0, 0.5)),
        #         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 1), per_channel=0.5),
        #         iaa.Affine(
        #             scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
        #             translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
        #             rotate=(-25, 25),
        #             shear=(-8, 8),
        #         ),
        #     ]),
        #     # iaa.CenterCropToFixedSize(*size)])
        #     iaa.CropToFixedSize(*size)])
        return iaa.Sequential([
            iaa.Sequential([
                iaa.Fliplr(0.5),  # horizontal flips,
                iaa.Rot90([0, 1, 2, 3]),
                iaa.LinearContrast((0.9, 1.1)),
                iaa.AdditiveGaussianNoise(scale=0.01),
                # iaa.Affine(rotate=(-10, 10)),
            ], random_order=True),
            iaa.CenterCropToFixedSize(*size)])

        # return iaa.Sequential([iaa.Fliplr(0.5),
        #                        iaa.Affine(rotate=(-25, 25)),
        #                        iaa.LinearContrast((0.5, 1.5)),
        #                        iaa.CenterCropToFixedSize(*size)])


def visualize_3d_patches(x1, x2, sample):
    import pylab as plt
    import matplotlib
    matplotlib.use('TkAgg')
    from skimage.util import montage

    def imshow(img, idx):
        plt.figure(idx, figsize=(img.shape[1] / 20 * 5, img.shape[0] / 20 * 4), frameon=False),
        plt.imshow(montage(img.transpose([-1, 0, 1]), grid_shape=(4, 5), padding_width=1, fill=-img.min()),
                   cmap='gray'),
        plt.axis('off'), plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    imshow(x1, 1)
    imshow(x2, 2)
    imshow(sample, 3)
    plt.show()
