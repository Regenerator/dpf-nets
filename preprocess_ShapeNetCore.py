import argparse
import os
import multiprocessing
import gc
import sys
import numpy as np
import pandas as pd
import h5py as h5

from lib.meshes.objmesh import ObjMesh


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Data processor for ShapeNetCore dataset. '
        'All OBJ files are preprocessed and accumulated in a single .h5 file.'
    )
    parser.add_argument('data', type=str, default='/hdd/data/ShapeNet_Core/',
                        help='Path to directory containing the unpacked dataset.')
    parser.add_argument('save', type=str, default='/hdd/data/ShapeNet_Core/',
                        help='Path to directory for the output.')

    return parser


def process_obj_file(sample):
    sample_obj = ObjMesh(sample)
    sample_obj.cleanup()
    data = sample_obj.reformat()

    del sample_obj
    gc.collect()

    return data


def process(part, cat2label, split, fout, args, n_workers=12, batch_size=1200):
    # Read filenames and labels #
    samples = []
    labels = []
    for i in range(len(split[split['split'] == part])):
        name = '0{}/{}/models/'.format(
            str(split[split['split'] == 'train']['synsetId'].values[i]),
            str(split[split['split'] == 'train']['modelId'].values[i])
        )
        if os.path.exists(os.path.join(args.data, 'shapes', name)):
            samples.append(name)
            labels.append(cat2label['0{}'.format(str(split[split['split'] == part]['synsetId'].values[i]))])
        else:
            print(name + ' does not exist!')

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
        processing_list = list(map(lambda s: os.path.join(args.data, 'shapes', s, 'model_normalized.obj'), samples[batch_size * b_i:batch_size * (b_i + 1)]))
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

        sys.stdout.write('Progress: [{}/{}]\n'.format(b_i + 1, n_batches))
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

    split = pd.read_csv(os.path.join(args.data, 'all.csv'))
    cat2label = {
        '0{}'.format(str(cat)): i for i, cat in enumerate(np.unique(split['synsetId'].values))
    }

    fout = h5.File(os.path.join(args.save, 'ShapeNetCore_tt.h5'), 'w')
    process('train', cat2label, split, fout, args)
    process('val', cat2label, split, fout, args)
    process('test', cat2label, split, fout, args)
    fout.close()


if __name__ == '__main__':
    main()
