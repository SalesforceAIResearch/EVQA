# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import torchvision.transforms as T

HIERA_MEAN = [0.485, 0.456, 0.406]
HIERA_STD = [0.229, 0.224, 0.225]


class Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        mean, std = video.new_tensor(self.mean), video.new_tensor(self.std)
        mean, std = mean[None, :, None, None], std[None, :, None, None]
        return (video - mean) / std


class Resize(T.Resize):

    def __init__(self, size):
        super().__init__(size, antialias=True)


class ToTensor:

    def __call__(self, video):
        return video.float().permute(0, 3, 1, 2) / 255


def get_sam2_transform(size):
    return T.Compose([ToTensor(), Resize((size, size)), Normalize(HIERA_MEAN, HIERA_STD)])
