import os
import h5py as h5
import numpy as np

from datetime import datetime
from torch.utils.data import Dataset

from .cloud_sampling import sample_cloud


class ShapeNetCoreDataset(Dataset):
    def __init__(self, path2data, part='train', meshes_fname='meshes.h5',
                 cloud_size=2**10, return_eval_cloud=False,
                 return_original_scale=False, return_bbox_scale=False,
                 cloud_transform=None,
                 sample_labels=False, chosen_label=None):
        super(ShapeNetCoreDataset, self).__init__()
        self.path2data = path2data
        self.meshes_fname = meshes_fname
        self.cloud_size = cloud_size
        self.return_eval_cloud = return_eval_cloud
        self.return_original_scale = return_original_scale
        self.return_bbox_scale = return_bbox_scale
        self.cloud_transform = cloud_transform
        self.sample_labels = sample_labels
        self.chosen_label = chosen_label

        self.data_file = None
        self.choose_part(part)

    def choose_part(self, part):
        self.part = part
        with h5.File(os.path.join(self.path2data, self.meshes_fname), 'r', libver='latest', swmr=True) as fin:
            if self.sample_labels:
                self.labels = np.zeros((fin[part + '_labels'].shape[0], 55), dtype=np.float32)
                self.labels[np.arange(self.labels.shape[0]), np.array(fin[part + '_labels'])] = 1.

            self.vertices_c_bounds = np.empty(fin[part + '_vertices_c_bounds'].shape, dtype=np.uint64)
            fin[part + '_vertices_c_bounds'].read_direct(self.vertices_c_bounds)

            self.faces_bounds = np.empty(fin[part + '_faces_bounds'].shape, dtype=np.uint64)
            fin[part + '_faces_bounds'].read_direct(self.faces_bounds)

            if self.return_original_scale:
                self.original_centers = np.empty(fin[part + '_orig_c'].shape, dtype=np.float32)
                fin[part + '_orig_c'].read_direct(self.original_centers)
                self.original_scales = np.empty(fin[part + '_orig_s'].shape, dtype=np.float32)
                fin[part + '_orig_s'].read_direct(self.original_scales)

            if self.return_bbox_scale:
                self.bbox_centers = np.empty(fin[part + '_bbox_c'].shape, dtype=np.float32)
                fin[part + '_bbox_c'].read_direct(self.bbox_centers)
                self.bbox_scales = np.empty(fin[part + '_bbox_s'].shape, dtype=np.float32)
                fin[part + '_bbox_s'].read_direct(self.bbox_scales)

            if self.chosen_label is not None:
                self.chosen_label_inds = (np.array(fin[part + '_labels'], dtype=np.uint8) == self.chosen_label).nonzero()[0]

    def close(self):
        if self.data_file is not None:
            self.data_file.close()

    def __len__(self):
        if self.chosen_label is not None:
            return self.chosen_label_inds.shape[0]
        else:
            return self.vertices_c_bounds.shape[0] - 1

    def __getitem__(self, i):
        np.random.seed(datetime.now().second + datetime.now().microsecond)

        if self.chosen_label is not None:
            i = self.chosen_label_inds[i]

        if self.data_file is None:
            self.data_file = h5.File(os.path.join(self.path2data, self.meshes_fname), 'r', libver='latest', swmr=True)

        vertices_c = np.array(
            self.data_file[self.part + '_vertices_c'][self.vertices_c_bounds[i]:self.vertices_c_bounds[i + 1]],
            dtype=np.float32
        )
        faces_vc = np.array(
            self.data_file[self.part + '_faces_vc'][self.faces_bounds[i]:self.faces_bounds[i + 1]],
            dtype=np.uint32
        )
        sample = sample_cloud(
            vertices_c, faces_vc,
            size=self.cloud_size,
            return_eval_cloud=self.return_eval_cloud
        )

        if self.return_original_scale:
            sample['orig_c'] = self.original_centers[i]
            sample['orig_s'] = self.original_scales[i]

        if self.return_bbox_scale:
            sample['bbox_c'] = self.bbox_centers[i]
            sample['bbox_s'] = self.bbox_scales[i]

        if self.cloud_transform is not None:
            sample = self.cloud_transform(sample)

        if self.sample_labels:
            sample['label'] = self.labels[i]

        return sample


class ShapeNetAllDataset(Dataset):
    def __init__(self, path2data, part='train',
                 images_fname='images.h5', meshes_fname='meshes.h5',
                 cloud_size=2**10, return_eval_cloud=False,
                 return_original_scale=False, return_bbox_scale=False,
                 image_transform=None, cloud_transform=None,
                 sample_labels=False, chosen_label=None):
        super(ShapeNetAllDataset, self).__init__()
        self.path2data = path2data
        self.images_fname = images_fname
        self.meshes_fname = meshes_fname
        self.cloud_size = cloud_size
        self.return_eval_cloud = return_eval_cloud
        self.return_original_scale = return_original_scale
        self.return_bbox_scale = return_bbox_scale
        self.image_transform = image_transform
        self.cloud_transform = cloud_transform
        self.sample_labels = sample_labels
        self.chosen_label = chosen_label

        self.images_file = None
        self.shapes_file = None
        self.choose_part(part)

    def choose_part(self, part):
        self.part = part
        with h5.File(os.path.join(self.path2data, self.meshes_fname), 'r', libver='latest', swmr=True) as fsh:
            if self.sample_labels:
                self.labels = np.zeros((fsh[part + '_labels'].shape[0], 55), dtype=np.float32)
                self.labels[np.arange(self.labels.shape[0]), np.array(fsh[part + '_labels'])] = 1.

            self.vertices_c_bounds = np.empty(fsh[part + '_vertices_c_bounds'].shape, dtype=np.uint64)
            fsh[part + '_vertices_c_bounds'].read_direct(self.vertices_c_bounds)

            self.faces_bounds = np.empty(fsh[part + '_faces_bounds'].shape, dtype=np.uint64)
            fsh[part + '_faces_bounds'].read_direct(self.faces_bounds)

            if self.return_original_scale:
                self.original_centers = np.empty(fsh[part + '_orig_c'].shape, dtype=np.float32)
                fsh[part + '_orig_c'].read_direct(self.original_centers)
                self.original_scales = np.empty(fsh[part + '_orig_s'].shape, dtype=np.float32)
                fsh[part + '_orig_s'].read_direct(self.original_scales)

            if self.return_bbox_scale:
                self.bbox_centers = np.empty(fsh[part + '_bbox_c'].shape, dtype=np.float32)
                fsh[part + '_bbox_c'].read_direct(self.bbox_centers)
                self.bbox_scales = np.empty(fsh[part + '_bbox_s'].shape, dtype=np.float32)
                fsh[part + '_bbox_s'].read_direct(self.bbox_scales)

            if self.chosen_label is not None:
                self.chosen_label_inds = (np.array(fsh[part + '_labels'], dtype=np.uint8) == self.chosen_label).nonzero()[0]

    def close(self):
        if self.shapes_file is not None:
            self.shapes_file.close()
        if self.images_file is not None:
            self.images_file.close()

    def __len__(self):
        if self.chosen_label is not None:
            return 24 * self.chosen_label_inds.shape[0]
        else:
            return 24 * (self.vertices_c_bounds.shape[0] - 1)

    def __getitem__(self, i):
        np.random.seed(datetime.now().second + datetime.now().microsecond)

        if self.chosen_label is not None:
            sh_i = self.chosen_label_inds[i // 24]
            im_i = 24 * self.chosen_label_inds[i // 24] + (i % 24)
        else:
            sh_i = i // 24
            im_i = i

        if self.shapes_file is None:
            self.images_file = h5.File(os.path.join(self.path2data, self.images_fname), 'r', libver='latest', swmr=True)
            self.shapes_file = h5.File(os.path.join(self.path2data, self.meshes_fname), 'r', libver='latest', swmr=True)

        vertices_c = np.array(
            self.shapes_file[self.part + '_vertices_c'][self.vertices_c_bounds[sh_i]:self.vertices_c_bounds[sh_i + 1]],
            dtype=np.float32
        )
        faces_vc = np.array(
            self.shapes_file[self.part + '_faces_vc'][self.faces_bounds[sh_i]:self.faces_bounds[sh_i + 1]],
            dtype=np.uint32
        )
        sample = sample_cloud(
            vertices_c, faces_vc,
            size=self.cloud_size,
            return_eval_cloud=self.return_eval_cloud
        )

        sample.update({
            'image': self.images_file[self.part + '_images'][im_i]
        })

        if self.return_original_scale:
            sample['orig_c'] = self.original_centers[sh_i]
            sample['orig_s'] = self.original_scales[sh_i]

        if self.return_bbox_scale:
            sample['bbox_c'] = self.bbox_centers[sh_i]
            sample['bbox_s'] = self.bbox_scales[sh_i]

        if self.image_transform is not None:
            sample['image'] = self.image_transform(sample['image'])

        if self.cloud_transform is not None:
            sample = self.cloud_transform(sample)

        if self.sample_labels:
            sample['label'] = self.labels[sh_i]

        return sample
