import os

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                  self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


# transforms

class ToGrayScale(object):
    """Convert a color image to grayscale"""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return {'image': image, 'keypoints': key_pts}


class Normalize(object):
    """Normalize the color range to [0,1]."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0

        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100) / 50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class RandomVFlip(object):
    """Randomly flip the image vertically with the specified probability"""

    def __init__(self, probability):
        self.probability = probability
        # remap the mirrored key points to the correct targets
        self.remapping = np.array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,  # cheeks
                                   26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  # brows
                                   27, 28, 29, 30,  # nose
                                   35, 34, 33, 32, 31,  # nostrils
                                   45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 40, 41,  # eyes
                                   54, 53, 52, 51, 50, 49, 48,  # upper lip
                                   59, 58, 57, 56, 55,  # lower lip
                                   64, 63, 62, 61, 60, 67, 66, 65])  # mouth

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        _, w = image.shape[:2]

        if np.random.rand() < self.probability:
            image = cv2.flip(image, 1)
            key_pts = np.array([[w - x, y] for x, y in key_pts])[self.remapping]

        return {'image': image, 'keypoints': key_pts}


class RandomContrastReduction(object):
    """
    Apply contrast reduction with the given probability and shift
    X = shift*X + (1 - shift) * mean(X)
    """

    def __init__(self, probability, shift):
        self.probability = probability
        self.shift = shift

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        if np.random.rand() < self.probability:
            mean = image.mean()
            image = image * self.shift + (1.0 - self.shift) * mean

        return {'image': image, 'keypoints': key_pts}


class RandomRot(object):
    """Rotate image randomly by the specified probability and angle [deg]"""

    def __init__(self, probability, angle):
        self.probability = probability
        self.angle = angle

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]

        if np.random.rand() < self.probability:
            mean = image.mean()
            a_deg = self.angle * np.random.choice([-1, 1])
            img_r = cv2.getRotationMatrix2D((w / 2, h / 2), a_deg, 1)
            image = cv2.warpAffine(image, img_r, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=mean)

            pts_r = self._calc_point_rotation_matrix(a_deg)
            center = [w / 2, h / 2]

            def point_rotation(pt):
                pt -= center
                pt = np.dot(pts_r, pt)
                return pt + center

            key_pts = np.array([point_rotation(pt) for pt in key_pts])

        return {'image': image, 'keypoints': key_pts}

    @staticmethod
    def _calc_point_rotation_matrix(a_deg):
        theta = np.radians(a_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, s), (-s, c)))


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # if image has no grayscale color channel, add one
        if (len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
