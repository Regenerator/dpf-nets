import argparse
import sys
import h5py as h5
import numpy as np


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Resampler for repacked ShapeNetCore dataset. '
        'Uses an .h5 dataset file to reshuffle train/val/test split and outputs another .h5 file.'
    )
    parser.add_argument('data_path', type=str, help='Path to repacked .h5 dataset file.')
    return parser


def process(orig_data, resampled_data, data_inds, part):
    resampled_data['{}_vertices_c_bounds'.format(part)][0] = 0
    resampled_data['{}_faces_bounds'.format(part)][0] = 0

    for i, ts in enumerate(data_inds):
        resampled_data['{}_vertices_c_bounds'.format(part)][i + 1] = (
            resampled_data['{}_vertices_c_bounds'.format(part)][i] +
            (orig_data['{}_vertices_c_bounds'.format(ts[0])][ts[1] + 1] -
             orig_data['{}_vertices_c_bounds'.format(ts[0])][ts[1]])
        )

        resampled_data['{}_vertices_c'.format(part)].resize(
            (resampled_data['{}_vertices_c_bounds'.format(part)][i + 1],) +
            resampled_data['{}_vertices_c'.format(part)].shape[1:]
        )
        resampled_data['{}_vertices_c'.format(part)][resampled_data['{}_vertices_c_bounds'.format(part)][i]:] = \
            orig_data['{}_vertices_c'.format(ts[0])][
                orig_data['{}_vertices_c_bounds'.format(ts[0])][ts[1]]:
                orig_data['{}_vertices_c_bounds'.format(ts[0])][ts[1] + 1]
        ]

        resampled_data['{}_faces_bounds'.format(part)][i + 1] = (
            resampled_data['{}_faces_bounds'.format(part)][i] +
            (orig_data['{}_faces_bounds'.format(ts[0])][ts[1] + 1] -
             orig_data['{}_faces_bounds'.format(ts[0])][ts[1]])
        )

        resampled_data['{}_faces_vc'.format(part)].resize(
            (resampled_data['{}_faces_bounds'.format(part)][i + 1],) +
            resampled_data['{}_faces_vc'.format(part)].shape[1:]
        )
        resampled_data['{}_faces_vc'.format(part)][resampled_data['{}_faces_bounds'.format(part)][i]:] = \
            orig_data['{}_faces_vc'.format(ts[0])][
                orig_data['{}_faces_bounds'.format(ts[0])][ts[1]]:
                orig_data['{}_faces_bounds'.format(ts[0])][ts[1] + 1]
        ]

        resampled_data['{}_orig_s'.format(part)][i] = orig_data['{}_orig_s'.format(ts[0])][ts[1]]
        resampled_data['{}_orig_c'.format(part)][i] = orig_data['{}_orig_c'.format(ts[0])][ts[1]]

        resampled_data['{}_bbox_s'.format(part)][i] = orig_data['{}_bbox_s'.format(ts[0])][ts[1]]
        resampled_data['{}_bbox_c'.format(part)][i] = orig_data['{}_bbox_c'.format(ts[0])][ts[1]]

        resampled_data['{}_labels'.format(part)][i] = orig_data['{}_labels'.format(ts[0])][ts[1]]

        sys.stdout.write('\r{}/{}'.format(i + 1, len(data_inds)))
        sys.stdout.flush()
    sys.stdout.write('\n')


def main():
    parser = define_options_parser()
    args = parser.parse_args()

    # Fix the seed #
    np.random.seed(seed=1)

    orig_data = h5.File(args.data_path, 'r')

    # Resample data inds #
    train_i = []
    val_i = []
    test_i = []
    for c in range(55):
        i2op = []
        i2oi = []
        c_tr_inds = (np.array(orig_data['train_labels']) == c).nonzero()[0]
        i2op += len(c_tr_inds) * ['train']
        i2oi += c_tr_inds.tolist()

        c_va_inds = (np.array(orig_data['val_labels']) == c).nonzero()[0]
        i2op += len(c_va_inds) * ['val']
        i2oi += c_va_inds.tolist()

        c_te_inds = (np.array(orig_data['test_labels']) == c).nonzero()[0]
        i2op += len(c_te_inds) * ['test']
        i2oi += c_te_inds.tolist()

        sample_i = np.arange(len(i2op))
        np.random.shuffle(sample_i)
        data = list(zip(list(map(lambda j: i2op[j], sample_i)), list(map(lambda j: i2oi[j], sample_i))))

        train_i += data[:len(c_tr_inds)]
        val_i += data[len(c_tr_inds):len(c_tr_inds) + len(c_va_inds)]
        test_i += data[len(c_tr_inds) + len(c_va_inds):]

    # Process data #
    resampled_data = h5.File(args.data_path[:-3] + '_resampled.h5', 'w')
    for key in orig_data.keys():
        if 'bounds' in key or 'labels' in key or 'orig' in key or 'bbox' in key:
            resampled_data.create_dataset(key, shape=orig_data[key].shape, dtype=orig_data[key].dtype)
        else:
            resampled_data.create_dataset(key, shape=(0,) + orig_data[key].shape[1:],
                                          maxshape=(None,) + orig_data[key].shape[1:], dtype=orig_data[key].dtype)

    process(orig_data, resampled_data, train_i, 'train')
    process(orig_data, resampled_data, val_i, 'val')
    process(orig_data, resampled_data, test_i, 'test')

    orig_data.close()
    resampled_data.close()


if __name__ == '__main__':
    main()
