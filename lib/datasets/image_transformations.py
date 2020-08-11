import cv2
import numpy as np

from torchvision.transforms import Compose


class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, image):
        img = np.float32(image / 255.)
        img[:3] = np.expand_dims(img[3], 0) * img[:3]
        return img


class Resize(object):
    def __init__(self, **kwargs):
        self.size = kwargs.get('image_size')

    def __call__(self, image):
        return np.transpose(cv2.resize(np.transpose(
            image, (1, 2, 0)
        ), (self.size[0], self.size[1])), (2, 0, 1))


class Pad(object):
    def __init__(self, **kwargs):
        self.pad_size = kwargs.get('image_pad_size')

    def __call__(self, image):
        padded = np.zeros((image.shape[0],
                           image.shape[1] + 2 * self.pad_size[0],
                           image.shape[2] + 2 * self.pad_size[1]), dtype=np.float32)
        padded[:, self.pad_size[0]:-self.pad_size[0], self.pad_size[1]:-self.pad_size[1]] = image
        return padded


class AddGrayscale(object):
    def __init__(self):
        self.r = 0.299
        self.g = 0.587
        self.b = 0.114

    def __call__(self, image):
        return np.vstack((
            np.expand_dims(self.r * image[0] + self.g * image[1] + self.b * image[2], 0), image
        ))


class NormalizeImages(object):
    def __init__(self, **kwargs):
        self.mean = np.array(kwargs.get('image_means'), dtype=np.float32)
        self.std = np.array(kwargs.get('image_stds'), dtype=np.float32)

    def __call__(self, image):
        return (image - self.mean.reshape(-1, 1, 1)) / self.std.reshape(-1, 1, 1)


class AddNoise2Images(object):
    def __init__(self, **kwargs):
        self.scale = kwargs.get('image_noise_scale')

    def __call__(self, image):
        return np.clip(image + np.float32(np.random.normal(scale=self.scale, size=image.shape)), 0.0, 1.0)


class RemoveAlpha(object):
    def __init__(self):
        pass

    def __call__(self, images):
        return images[:4]


def ComposeImageTransformation(**kwargs):
    image_transformations = []
    image_transformations.append(ToNumpy())
    if kwargs.get('image_resize'):
        image_transformations.append(Resize(**kwargs))
    if kwargs.get('image_pad'):
        image_transformations.append(Pad(**kwargs))
    if kwargs.get('image_add_grayscale'):
        image_transformations.append(AddGrayscale())
    if kwargs.get('image_normalize'):
        image_transformations.append(NormalizeImages(**kwargs))
    if kwargs.get('image_noise'):
        image_transformations.append(AddNoise2Images(**kwargs))
    if kwargs.get('image_remove_alpha'):
        image_transformations.append(RemoveAlpha())

    if len(image_transformations) == 0:
        return None
    else:
        return Compose(image_transformations)
