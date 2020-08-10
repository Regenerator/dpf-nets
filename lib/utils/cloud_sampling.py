import numpy as np


def sample_cloud(vertices_c, faces_vc, size=2**10, return_eval_cloud=False):
    polygons = vertices_c[faces_vc]
    cross = np.cross(polygons[:, 2] - polygons[:, 0], polygons[:, 2] - polygons[:, 1])
    areas = np.sqrt((cross**2).sum(1)) / 2.0

    probs = areas / areas.sum()
    p_sample = np.random.choice(np.arange(polygons.shape[0]), size=2 * size if return_eval_cloud else size, p=probs)

    sampled_polygons = polygons[p_sample]

    s1 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
    s2 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
    cond = (s1 + s2) > 1.
    s1[cond] = 1. - s1[cond]
    s2[cond] = 1. - s2[cond]

    sample = {
        'cloud': (sampled_polygons[:, 0] +
                  s1 * (sampled_polygons[:, 1] - sampled_polygons[:, 0]) +
                  s2 * (sampled_polygons[:, 2] - sampled_polygons[:, 0])).astype(np.float32)
    }

    if return_eval_cloud:
        sample['eval_cloud'] = sample['cloud'][1::2].copy().T
        sample['cloud'] = sample['cloud'][::2].T
    else:
        sample['cloud'] = sample['cloud'].T

    return sample
