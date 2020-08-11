import numpy as np

from torchvision.transforms import Compose


class Scale2OrigCloud(object):
    def __init__(self, **kwargs):
        self.do_rescale = kwargs['cloud_rescale2orig']
        self.do_recenter = kwargs['cloud_recenter2orig']

    def __call__(self, sample):
        if self.do_rescale:
            sample['cloud'] = sample['orig_s'] * sample['cloud']
            if 'eval_cloud' in sample:
                sample['eval_cloud'] = sample['orig_s'] * sample['eval_cloud']
        if self.do_recenter:
            sample['cloud'] = sample['cloud'] + sample['orig_c'].reshape(-1, 1)
            if 'eval_cloud' in sample:
                sample['eval_cloud'] = sample['eval_cloud'] + sample['orig_c'].reshape(-1, 1)
        return sample


class TranslateCloud(object):
    def __init__(self, **kwargs):
        self.shift = np.array(kwargs['cloud_translate_shift'], dtype=np.float32).reshape(-1, 1)

    def __call__(self, sample):
        sample['cloud'] -= self.shift
        if 'eval_cloud' in sample:
            sample['eval_cloud'] -= self.shift
        return sample


class ScaleCloud(object):
    def __init__(self, **kwargs):
        self.scale = np.float32(kwargs.get('cloud_scale_scale'))

    def __call__(self, sample):
        sample['cloud'] /= self.scale
        if 'eval_cloud' in sample:
            sample['eval_cloud'] /= self.scale
        return sample


class AddNoise2Cloud(object):
    def __init__(self, **kwargs):
        self.scale = np.float32(kwargs.get('cloud_noise_scale'))

    def __call__(self, sample):
        sample['cloud'] += np.random.normal(scale=self.scale, size=sample['cloud'].shape).astype(np.float32)
        if 'eval_cloud' in sample:
            sample['eval_cloud'] += np.random.normal(scale=self.scale, size=sample['eval_cloud'].shape).astype(np.float32)
        return sample


class CenterCloud(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['cloud'] -= sample['cloud'].mean(axis=1, keepdims=True)
        if 'eval_cloud' in sample:
            sample['eval_cloud'] -= sample['eval_cloud'].mean(axis=1, keepdims=True)
        return sample


def ComposeCloudTransformation(**kwargs):
    cloud_transformation = []
    if kwargs.get('cloud_rescale2orig') or kwargs.get('cloud_recenter2orig'):
        cloud_transformation.append(Scale2OrigCloud(**kwargs))
    if kwargs.get('cloud_translate'):
        cloud_transformation.append(TranslateCloud(**kwargs))
    if kwargs.get('cloud_scale'):
        cloud_transformation.append(ScaleCloud(**kwargs))
    if kwargs.get('cloud_noise'):
        cloud_transformation.append(AddNoise2Cloud(**kwargs))
    if kwargs.get('cloud_center'):
        cloud_transformation.append(CenterCloud())

    if len(cloud_transformation) == 0:
        return None
    else:
        return Compose(cloud_transformation)
