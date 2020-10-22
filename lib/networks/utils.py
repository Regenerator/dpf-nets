import torch
import numpy as np
from scipy.stats import entropy

from lib.metrics.StructuralLosses.nn_distance import nn_distance


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(state, model_name):
    torch.save(state, model_name, pickle_protocol=4)
    print('Model saved to ' + model_name)


def cnt_params(params):
    return sum(p.numel() for p in params if p.requires_grad)


def distChamferCUDA(x, y):
    return nn_distance(x, y)


def f_score(predicted_clouds, true_clouds, threshold=0.001):
    ld, rd = distChamferCUDA(predicted_clouds, true_clouds)
    precision = 100. * (rd < threshold).float().mean(1)
    recall = 100. * (ld < threshold).float().mean(1)
    return 2. * precision * recall / (precision + recall + 1e-7)


def get_voxel_occ_dist(all_clouds, clouds_flag='gen', res=28, bound=0.5, bs=128, warning=True):
    if np.any(np.fabs(all_clouds) > bound) and warning:
        print('{} clouds out of cube bounds: [-{}; {}]'.format(clouds_flag, bound, bound))

    n_nans = np.isnan(all_clouds).sum()
    if n_nans > 0:
        print('{} NaN values in point cloud tensors.'.format(n_nans))

    p2v_dist = np.zeros((res, res, res), dtype=np.uint64)

    step = 1. / res
    v_bs = -0.5 + np.arange(res + 1) * step

    nbs = all_clouds.shape[0] // bs + 1
    for i in range(nbs):
        clouds = all_clouds[bs * i:bs * (i + 1)]

        preiis = clouds[:, :, 0].reshape(1, -1)
        preiis = np.logical_and(v_bs[:28].reshape(-1, 1) <= preiis, preiis < v_bs[1:].reshape(-1, 1))
        iis = preiis.argmax(0)
        iis_values = preiis.sum(0) > 0

        prejjs = clouds[:, :, 1].reshape(1, -1)
        prejjs = np.logical_and(v_bs[:28].reshape(-1, 1) <= prejjs, prejjs < v_bs[1:].reshape(-1, 1))
        jjs = prejjs.argmax(0)
        jjs_values = prejjs.sum(0) > 0

        prekks = clouds[:, :, 2].reshape(1, -1)
        prekks = np.logical_and(v_bs[:28].reshape(-1, 1) <= prekks, prekks < v_bs[1:].reshape(-1, 1))
        kks = prekks.argmax(0)
        kks_values = prekks.sum(0) > 0

        values = np.uint64(np.logical_and(np.logical_and(iis_values, jjs_values), kks_values))
        np.add.at(p2v_dist, (iis, jjs, kks), values)

    return np.float64(p2v_dist) / p2v_dist.sum()


def JSD(clouds1, clouds2, clouds1_flag='gen', clouds2_flag='ref', warning=True):
    dist1 = get_voxel_occ_dist(clouds1, clouds_flag=clouds1_flag, warning=warning)
    dist2 = get_voxel_occ_dist(clouds2, clouds_flag=clouds2_flag, warning=warning)
    return entropy((dist1 + dist2).flatten() / 2.0, base=2) - \
        0.5 * (entropy(dist1.flatten(), base=2) + entropy(dist2.flatten(), base=2))


def pairwise_CD(clouds1, clouds2, bs=2048):
    N1 = clouds1.shape[0]
    N2 = clouds2.shape[0]

    cds = torch.from_numpy(np.zeros((N1, N2), dtype=np.float32)).cuda()

    for i in range(N1):
        clouds1_i = clouds1[i]

        if bs < N1:
            for j_l in range(0, N2, bs):
                j_u = min(N2, j_l + bs)
                clouds2_js = clouds2[j_l:j_u]

                clouds1_is = clouds1_i.unsqueeze(0).expand(j_u - j_l, -1, -1)
                clouds1_is = clouds1_is.contiguous()

                dl, dr = distChamferCUDA(clouds1_is, clouds2_js)
                cds[i, j_l:j_u] = dl.mean(dim=1) + dr.mean(dim=1)

        else:
            clouds1_is = clouds1_i.unsqueeze(0).expand(N1, -1, -1)
            clouds1_is = clouds1_is.contiguous()

            dl, dr = distChamferCUDA(clouds1_is, clouds2)
            cds[i] = dl.mean(dim=1) + dr.mean(dim=1)

    return cds


def COV(dists, axis=1):
    return float(dists.min(axis)[1].unique().shape[0]) / float(dists.shape[axis])


def MMD(dists, axis=1):
    return float(dists.min((axis + 1) % 2)[0].mean().float())


def KNN(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((-torch.ones(n0), torch.ones(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, 0).float()
    pred[torch.eq(pred, 0)] = -1.

    return float(torch.eq(label, pred).float().mean())
