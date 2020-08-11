import argparse
import os
import shutil
import multiprocessing
import gc
import sys
import cv2
import numpy as np
import h5py as h5

from itertools import product
from lib.meshes.objmesh import ObjMesh


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Data processor for ShapeNetCore dataset. '
        'All OBJ files are preprocessed and accumulated in a single .h5 file.'
    )
    parser.add_argument('snc1_data_dir', type=str, help='Path to directory containing the unpacked ShapeNetCore.v1 dataset.')
    parser.add_argument('sna_data_dir', type=str, help='Path to directory containing the unpacked ShapeNetAll dataset.')
    parser.add_argument('save_dir', type=str, help='Path to directory for the output.')
    parser.add_argument('n_processes', type=int, help='Number of parallel processing jobs.')
    parser.add_argument('batch_size', type=int, help='Number of shapes in processed batches.')
    return parser


def process_png_file(sample):
    img = np.expand_dims(np.transpose(np.array(cv2.imread(sample), dtype=np.uint8), (2, 0, 1)), 0)
    return img


def process_obj_file(sample):
    sample_obj = ObjMesh(sample)
    sample_obj.cleanup()
    data = sample_obj.reformat()

    del sample_obj
    gc.collect()

    return data


def process_images(part, cats, cat2label, fout, args, n_workers=12, batch_size=100):
    # Read filenames #
    samples = []
    labels = []
    for cat in cats:
        cat_names = sorted([
            name for name in os.listdir(os.path.join(args.sna_data_dir, 'ShapeNetMesh', cat))
            if os.path.isdir(os.path.join(args.sna_data_dir, 'ShapeNetMesh', cat, name))
        ])
        cat_size = len(cat_names)
        if part == 'train':
            cat_names = cat_names[:int(0.8 * cat_size)]
        elif part == 'test':
            cat_names = cat_names[int(0.8 * cat_size):]

        samples += list(map(
            lambda n: os.path.join(args.sna_data_dir, 'ShapeNetRendering', cat, n), cat_names
        ))
        labels += len(cat_names) * [cat2label[cat]]

    # Create datasets #
    images_ds = fout.create_dataset('{}_images'.format(part), shape=(24 * len(samples), 4, 137, 137), dtype=np.uint8)
    labels_ds = fout.create_dataset('{}_labels'.format(part), data=np.array(labels, dtype=np.uint8))

    # Read in batches #
    processing_pool = multiprocessing.Pool(processes=n_workers)
    n_batches = np.ceil(len(samples) / batch_size).astype(np.uint32)
    for b_i in range(n_batches):
        processing_list = list(map(
            lambda s: os.path.join(s[0], 'rendering', '{:02d}.png'.format(s[1])),
            product(samples[batch_size * b_i:batch_size * (b_i + 1)], np.arange(24))
        ))
        processing_results = processing_pool.map(process_png_file, processing_list)

        images_ds[24 * batch_size * b_i:24 * batch_size * (b_i + 1)] = np.concatenate(processing_results, 0)

        del processing_results
        gc.collect()

        sys.stdout.write('Packing {} images: [{}/{}]\n'.format(part, b_i + 1, n_batches))
        sys.stdout.flush()
    processing_pool.close()


def process_meshes(part, cats, cat2label, fout, args, n_workers=12, batch_size=1200):
    # Read filenames and labels #
    samples = []
    labels = []
    for cat in cats:
        cat_names = sorted([
            name for name in os.listdir(os.path.join(args.sna_data_dir, 'ShapeNetMesh', cat))
            if os.path.isdir(os.path.join(args.sna_data_dir, 'ShapeNetMesh', cat, name))
        ])
        cat_size = len(cat_names)
        if part == 'train':
            cat_names = cat_names[:int(0.8 * cat_size)]
        elif part == 'test':
            cat_names = cat_names[int(0.8 * cat_size):]

        samples += list(map(
            lambda n: os.path.join(args.sna_data_dir, 'ShapeNetMesh', cat, n), cat_names
        ))
        labels += len(cat_names) * [cat2label[cat]]

    # Create datasets #
    vcb_ds = fout.create_dataset('{}_vertices_c_bounds'.format(part), shape=(len(samples) + 1,), dtype=np.uint64)
    vcb_ds[0] = 0
    vc_ds = fout.create_dataset('{}_vertices_c'.format(part), shape=(0, 3), maxshape=(None, 3), dtype=np.float32)

    orig_c_ds = fout.create_dataset('{}_orig_c'.format(part), shape=(len(samples), 3), dtype=np.float32)
    orig_s_ds = fout.create_dataset('{}_orig_s'.format(part), shape=(len(samples),), dtype=np.float32)

    bbox_c_ds = fout.create_dataset('{}_bbox_c'.format(part), shape=(len(samples), 3), dtype=np.float32)
    bbox_s_ds = fout.create_dataset('{}_bbox_s'.format(part), shape=(len(samples),), dtype=np.float32)

    fb_ds = fout.create_dataset('{}_faces_bounds'.format(part), shape=(len(samples) + 1,), dtype=np.uint64)
    fb_ds[0] = 0
    fvc_ds = fout.create_dataset('{}_faces_vc'.format(part), shape=(0, 3), maxshape=(None, 3), dtype=np.uint32)

    labels_ds = fout.create_dataset('{}_labels'.format(part), data=np.array(labels, dtype=np.uint8))

    # Read in batches #
    processing_pool = multiprocessing.Pool(processes=n_workers)
    n_batches = np.ceil(len(samples) / batch_size).astype(np.uint32)
    for b_i in range(n_batches):
        processing_list = list(map(
            lambda s: os.path.join(s, 'model.obj'),
            samples[batch_size * b_i:batch_size * (b_i + 1)]
        ))
        processing_results = processing_pool.map(process_obj_file, processing_list)

        vcb_ds[batch_size * b_i + 1:batch_size * (b_i + 1) + 1] = \
            np.array(list(map(lambda d: len(d['vertices_c']), processing_results)), dtype=np.uint64).dot(
                np.triu(np.ones((len(processing_results), len(processing_results)), dtype=np.uint64))
        )

        b_vc = np.concatenate(list(map(lambda d: d['vertices_c'], processing_results)), axis=0)
        vc_ds_s = vc_ds.shape[0]
        vc_ds.resize((vc_ds_s + len(b_vc), 3))
        vc_ds[vc_ds_s:] = b_vc

        orig_c_ds[batch_size * b_i:batch_size * (b_i + 1)] = \
            np.concatenate(list(map(lambda d: d['orig_c'].reshape(1, -1), processing_results)), axis=0)
        orig_s_ds[batch_size * b_i:batch_size * (b_i + 1)] = \
            np.array(list(map(lambda d: d['orig_s'], processing_results)))

        bbox_c_ds[batch_size * b_i:batch_size * (b_i + 1)] = \
            np.concatenate(list(map(lambda d: d['bbox_c'].reshape(1, -1), processing_results)), axis=0)
        bbox_s_ds[batch_size * b_i:batch_size * (b_i + 1)] = \
            np.array(list(map(lambda d: d['bbox_s'], processing_results)))

        fb_ds[batch_size * b_i + 1:batch_size * (b_i + 1) + 1] = \
            np.array(list(map(lambda d: len(d['faces_vc']), processing_results)), dtype=np.uint64).dot(
                np.triu(np.ones((len(processing_results), len(processing_results)), dtype=np.uint64))
        )
        b_fvc = np.concatenate(list(map(lambda d: d['faces_vc'], processing_results)), axis=0)
        fv_ds_s = fvc_ds.shape[0]
        fvc_ds.resize((fv_ds_s + len(b_fvc), 3))
        fvc_ds[fv_ds_s:] = b_fvc

        del processing_results
        gc.collect()

        sys.stdout.write('Packing {} meshes: [{}/{}]\n'.format(part, b_i + 1, n_batches))
        sys.stdout.flush()
    processing_pool.close()

    # Repair cross batch shape vertices bounds #
    vcb = np.array(vcb_ds[:])
    vcb_upd = np.tile(
        np.tril(np.ones((n_batches, n_batches), dtype=np.uint64)).dot(vcb[0::batch_size]).reshape(-1, 1),
        (1, batch_size)
    ).flatten()[:(len(vcb) - 1)]
    vcb[1:] = vcb[1:] + vcb_upd
    vcb_ds[:] = vcb

    # Repair cross batch shape faces bounds #
    fb = np.array(fb_ds[:])
    fb_upd = np.tile(
        np.tril(np.ones((n_batches, n_batches), dtype=np.uint64)).dot(fb[0::batch_size]).reshape(-1, 1),
        (1, batch_size)
    ).flatten()[:(len(fb) - 1)]
    fb[1:] = fb[1:] + fb_upd
    fb_ds[:] = fb


def main():
    parser = define_options_parser()
    args = parser.parse_args()

    # Copy meshes from ShapeNetCore.v1 corresponding to shapes in ShapeNetAll #
    cats_all = sorted(os.listdir(os.path.join(args.sna_data_dir, 'ShapeNetVox32')))
    cats2samples = {
        cat: sorted(os.listdir(os.path.join(args.sna_data_dir, 'ShapeNetVox32', cat))) for cat in cats_all
    }

    for cat, samples in cats2samples.items():
        for sample in samples:
            samplename = os.path.join(cat, sample)
            shutil.copytree(os.path.join(args.snc1_data_dir, samplename),
                            os.path.join(args.sna_data_dir, 'ShapeNetMesh', samplename))

    cats = sorted(os.listdir(os.path.join(args.sna_data_dir, 'ShapeNetMesh')))
    cat2label = {
        '{}'.format(str(cat)): i for i, cat in enumerate(cats)
    }

    fout_images = h5.File(os.path.join(args.save_dir, 'ShapeNetAll13_images.h5'), 'w')
    process_images('train', cats, cat2label, fout_images, args, n_workers=args.n_processes, batch_size=args.batch_size // 24)
    process_images('test', cats, cat2label, fout_images, args, n_workers=args.n_processes, batch_size=args.batch_size // 24)
    fout_images.close()

    fout_meshes = h5.File(os.path.join(args.save_dir, 'ShapeNetAll13_meshes.h5'), 'w')
    process_meshes('train', cats, cat2label, fout_meshes, args, n_workers=args.n_processes, batch_size=args.batch_size)
    process_meshes('test', cats, cat2label, fout_meshes, args, n_workers=args.n_processes, batch_size=args.batch_size)
    fout_meshes.close()


if __name__ == '__main__':
    main()
