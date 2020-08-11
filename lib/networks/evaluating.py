import os
from time import time
from sys import stdout
from scipy.stats import entropy
from skimage import morphology

import h5py as h5
import numpy as np
import torch

from metrics.StructuralLosses.nn_distance import nn_distance


def distChamferCUDA(x, y):
    return nn_distance(x, y)


def f_score(predicted_clouds, true_clouds, threshold=0.001):
    ld, rd = distChamferCUDA(predicted_clouds, true_clouds)
    precision = 100. * (rd < threshold).float().mean(1)
    recall = 100. * (ld < threshold).float().mean(1)
    return 2. * precision * recall / (precision + recall + 1e-7)


def get_voxel_occupances(clouds, res=32, bound=0.5):
    step = 1. / res
    v_bs = -0.5 + np.arange(res + 1) * step

    surface_grids = np.zeros((clouds.shape[0], res, res, res), dtype=np.uint8)

    preiis = np.expand_dims(clouds[:, :, 0], 0)
    preiis = np.logical_and(v_bs[:res].reshape(-1, 1, 1) <= preiis, preiis < v_bs[1:].reshape(-1, 1, 1))
    iis = preiis.argmax(0).flatten()
    iis_is_inside = preiis.sum(0) > 0

    prejjs = np.expand_dims(clouds[:, :, 1], 0)
    prejjs = np.logical_and(v_bs[:res].reshape(-1, 1, 1) <= prejjs, prejjs < v_bs[1:].reshape(-1, 1, 1))
    jjs = prejjs.argmax(0).flatten()
    jjs_is_inside = prejjs.sum(0) > 0

    prekks = np.expand_dims(clouds[:, :, 2], 0)
    prekks = np.logical_and(v_bs[:res].reshape(-1, 1, 1) <= prekks, prekks < v_bs[1:].reshape(-1, 1, 1))
    kks = prekks.argmax(0).flatten()
    kks_is_inside = prekks.sum(0) > 0

    pc_inds = np.tile(np.arange(clouds.shape[0]).reshape(-1, 1), (1, clouds.shape[1])).flatten()
    pc_is_inside = np.logical_and(np.logical_and(iis_is_inside, jjs_is_inside), kks_is_inside).flatten()

    pc_inds = pc_inds[pc_is_inside]
    iis = iis[pc_is_inside]
    jjs = jjs[pc_is_inside]
    kks = kks[pc_is_inside]

    np.add.at(surface_grids, (pc_inds, iis, jjs, kks), 1)

    filled_grids = np.zeros((clouds.shape[0], res, res, res), dtype=np.uint8)
    for j in range(clouds.shape[0]):
        tmp_grid = np.zeros((res + 2, res + 2, res + 2), dtype=np.uint8)
        tmp_grid[1:-1, 1:-1, 1:-1] = surface_grids[j]
        labels, num_labels = morphology.label(tmp_grid, background=1, connectivity=1, return_num=True)
        outside_label = np.array([labels[0, 0, 0]])
        filled_grids[j][(labels != outside_label.reshape(1, 1, 1))[1:-1, 1:-1, 1:-1]] = 1

    return filled_grids


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
    torch.save(state, model_name)
    print('Model saved to ' + model_name)


def cnt_params(params):
    return sum(p.numel() for p in params if p.requires_grad)


def evaluate(iterator, model, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    saving_mode = kwargs.get('saving_mode')

    if saving_mode:
        clouds_fname = '{}_{}_{}_{}_clouds_{}.h5'.format(kwargs['model_name'][:-4],
                                                         iterator.dataset.part,
                                                         kwargs['cloud_size'],
                                                         kwargs['sampled_cloud_size'],
                                                         kwargs['usage_mode'])
        clouds_fname = os.path.join(kwargs['path2save'], clouds_fname)
        clouds_file = h5.File(clouds_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(kwargs['N_samples'] * len(iterator.dataset), 3,
                   kwargs['cloud_size'] if kwargs['refine_sampled_clouds'] else kwargs['sampled_cloud_size']),
            dtype=np.float32
        )
        gt_clouds = clouds_file.create_dataset(
            'gt_clouds',
            shape=(kwargs['N_samples'] * len(iterator.dataset), 3,
                   kwargs['cloud_size']),
            dtype=np.float32
        )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    inf_time = AverageMeter()

    if kwargs.get('usage_mode') == 'training':
        LB = AverageMeter()
        PNLL = AverageMeter()
        GNLL = AverageMeter()
        GENT = AverageMeter()

    elif kwargs.get('usage_mode') == 'evaluating':
        CD = AverageMeter()

    elif kwargs.get('usage_mode') == 'generating':
        gen_clouds_buf = []
        ref_clouds_buf = []

    elif kwargs.get('usage_mode') == 'predicting':
        CD = AverageMeter()
        F1 = AverageMeter()

    model.eval()
    torch.set_grad_enabled(False)

    end = time()

    for i, batch in enumerate(iterator):
        data_time.update(time() - end)

        clouds = batch['cloud'].cuda(non_blocking=True)
        ref_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        if 'ic' in train_mode:
            images = batch['image'].cuda(non_blocking=True)
            inf_end = time()
            outputs = model(clouds, ref_clouds, images, n_sampled_points=kwargs['sampled_cloud_size'])
        else:
            inf_end = time()
            outputs = model(clouds, ref_clouds, n_sampled_points=kwargs['sampled_cloud_size'])
        inf_time.update((time() - inf_end) / clouds.shape[0], clouds.shape[0])

        if kwargs.get('usage_mode') == 'training':
            loss, pnll, gnll, gent = loss_func(clouds, ref_clouds, outputs)
            LB.update((pnll + gnll - gent).item(), clouds.shape[0])
            PNLL.update(pnll.item(), clouds.shape[0])
            GNLL.update(gnll.item(), clouds.shape[0])
            GENT.update(gent.item(), clouds.shape[0])

        elif kwargs.get('usage_mode') == 'evaluating':
            rec_clouds = outputs['p_prior_samples'][-1]

            if kwargs['orig_scale_evaluation']:
                if kwargs['cloud_scale']:
                    rec_clouds *= kwargs['cloud_scale_scale']
                    ref_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    rec_clouds += shift
                    ref_clouds += shift

                if not kwargs['cloud_rescale2orig']:
                    rec_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    ref_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                if not kwargs['cloud_recenter2orig']:
                    rec_clouds += batch['orig_c'].unsqueeze(2).cuda()
                    ref_clouds += batch['orig_c'].unsqueeze(2).cuda()

            if saving_mode:
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + rec_clouds.shape[0]] = \
                    rec_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + ref_clouds.shape[0]] = \
                    ref_clouds.cpu().numpy().astype(np.float32)

            dl, dr = distChamferCUDA(torch.transpose(rec_clouds, 1, 2).contiguous(),
                                     torch.transpose(ref_clouds, 1, 2).contiguous())
            cd = (dl.mean(1) + dr.mean(1)).mean()
            CD.update(cd.item(), ref_clouds.shape[0])

        elif kwargs.get('usage_mode') == 'generating':
            gen_clouds = outputs['p_prior_samples'][-1]

            if kwargs['orig_scale_evaluation']:
                if kwargs['cloud_scale']:
                    gen_clouds *= kwargs['cloud_scale_scale']
                    ref_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift'],
                                                      dtype=np.float32).reshape(1, -1, 1)).cuda()
                    gen_clouds += shift
                    ref_clouds += shift

                if not kwargs['cloud_rescale2orig']:
                    gen_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    ref_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                if not kwargs['cloud_recenter2orig']:
                    gen_clouds += batch['orig_c'].unsqueeze(2).cuda()
                    ref_clouds += batch['orig_c'].unsqueeze(2).cuda()

            gen_clouds_buf.append(gen_clouds)
            ref_clouds_buf.append(ref_clouds)

            if saving_mode:
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + gen_clouds.shape[0]] = \
                    gen_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + ref_clouds.shape[0]] = \
                    ref_clouds.cpu().numpy().astype(np.float32)

        elif kwargs.get('usage_mode') == 'predicting':
            rec_clouds = outputs['p_prior_samples'][-1]

            if kwargs['unit_scale_evaluation']:
                if kwargs['cloud_scale']:
                    rec_clouds *= kwargs['cloud_scale_scale']
                    ref_clouds *= kwargs['cloud_scale_scale']

            if kwargs['orig_scale_evaluation']:
                if kwargs['cloud_scale']:
                    rec_clouds *= kwargs['cloud_scale_scale']
                    ref_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    rec_clouds += shift
                    ref_clouds += shift

                if not kwargs['cloud_rescale2orig']:
                    rec_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    ref_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                if not kwargs['cloud_recenter2orig']:
                    rec_clouds += batch['orig_c'].unsqueeze(2).cuda()
                    ref_clouds += batch['orig_c'].unsqueeze(2).cuda()

            if kwargs['bbox_scale_evaluation']:
                if kwargs['cloud_scale']:
                    rec_clouds *= kwargs['cloud_scale_scale']
                    ref_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    rec_clouds += shift
                    ref_clouds += shift

                if kwargs['cloud_recenter2orig']:
                    rec_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                    ref_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                if kwargs['cloud_rescale2orig']:
                    rec_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    ref_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()

                rec_clouds -= batch['bbox_c'].unsqueeze(2).cuda()
                ref_clouds -= batch['bbox_c'].unsqueeze(2).cuda()

                rec_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()
                ref_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()

            if saving_mode:
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + rec_clouds.shape[0]] = \
                    rec_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + ref_clouds.shape[0]] = \
                    ref_clouds.cpu().numpy().astype(np.float32)

            rec_clouds = torch.transpose(rec_clouds, 1, 2)
            ref_clouds = torch.transpose(ref_clouds, 1, 2)

            dl, dr = distChamferCUDA(rec_clouds[:, :kwargs['eval_cloud_size']].contiguous(),
                                     ref_clouds[:, :kwargs['eval_cloud_size']].contiguous())
            cd = (dl.mean(1) + dr.mean(1)).mean()
            f1 = f_score(rec_clouds[:, :kwargs['eval_cloud_size']].contiguous(),
                         ref_clouds[:, :kwargs['eval_cloud_size']].contiguous()).mean()
            CD.update(cd.item(), ref_clouds.shape[0])
            F1.update(f1.item(), ref_clouds.shape[0])

        batch_time.update(time() - end)
        line = '[{cc}/{il}]  \tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(cc=i + 1, il=len(iterator), batch_time=batch_time)
        if kwargs.get('usage_mode') == 'training':
            line += '\tLB {LB.val:.2f} ({LB.avg:.2f})'.format(LB=LB)
            line += '\tPNLL {PNLL.val:.2f} ({PNLL.avg:.2f})'.format(PNLL=PNLL)
            line += '\tGNLL {GNLL.val:.2f} ({GNLL.avg:.2f})'.format(GNLL=GNLL)
            line += '\tGENT {GENT.val:.2f} ({GENT.avg:.2f})'.format(GENT=GENT)

        elif kwargs.get('usage_mode') == 'evaluating':
            line += '\tCD {CD.val:.6f} ({CD.avg:.6f})'.format(CD=CD)

        elif kwargs.get('usage_mode') == 'predicting':
            line += '\tCD {CD.val:.6f} ({CD.avg:.6f})'.format(CD=CD)
            line += '\tF1 {F1.val:.1f} ({F1.avg:.1f})'.format(F1=F1)

        line += '\n'
        stdout.write(line)
        stdout.flush()
        end = time()

    print('Inference time: {} sec/sample'.format(inf_time.avg))

    if kwargs.get('usage_mode') == 'evaluating':
        print('CD: {:.6f}'.format(CD.avg))

    elif kwargs.get('usage_mode') == 'predicting':
        print('CD: {:.6f}'.format(CD.avg))
        print('F1: {:.1f}'.format(F1.avg))

    elif kwargs.get('usage_mode') == 'generating':
        gen_clouds_buf = torch.transpose(torch.cat(gen_clouds_buf, dim=0), 2, 1).contiguous()
        ref_clouds_buf = torch.transpose(torch.cat(ref_clouds_buf, dim=0), 2, 1).contiguous()

        gen_clouds_buf = gen_clouds_buf.cpu().numpy()
        gen_clouds_inds = set(np.arange(gen_clouds_buf.shape[0]))
        nan_gen_clouds_inds = set(np.isnan(gen_clouds_buf).sum(axis=(1, 2)).nonzero()[0])
        gen_clouds_inds = list(gen_clouds_inds - nan_gen_clouds_inds)
        dup_gen_clouds_inds = np.random.choice(gen_clouds_inds, size=len(nan_gen_clouds_inds))
        gen_clouds_buf[list(nan_gen_clouds_inds)] = gen_clouds_buf[dup_gen_clouds_inds]
        gen_clouds_buf = torch.from_numpy(gen_clouds_buf).cuda()

        gg_cds = pairwise_CD(gen_clouds_buf, gen_clouds_buf)
        tt_cds = pairwise_CD(ref_clouds_buf, ref_clouds_buf)
        gt_cds = pairwise_CD(gen_clouds_buf, ref_clouds_buf)

        jsd = JSD(gen_clouds_buf.cpu().numpy(), ref_clouds_buf.cpu().numpy(),
                  clouds1_flag='gen', clouds2_flag='ref', warning=False)
        cd_covs = COV(gt_cds)
        cd_mmds = MMD(gt_cds)
        cd_1nns = KNN(gg_cds, gt_cds, tt_cds, 1)

        print('JSD:   \t{:.2f}'.format(1e2 * jsd))
        print('COV-CD:\t{:.1f}'.format(1e2 * cd_covs))
        print('MMD-CD:\t{:.2f}'.format(1e4 * cd_mmds))
        print('1NN-CD:\t{:.1f}'.format(1e2 * cd_1nns))

    if saving_mode and kwargs['N_samples'] > 1:
        for i in range(1, kwargs['N_samples'], 1):
            for j, batch in enumerate(iterator):
                clouds = batch['cloud'].cuda(non_blocking=True)
                ref_clouds = batch['eval_cloud'].cuda(non_blocking=True)
                outputs = model(clouds, ref_clouds, n_sampled_points=kwargs['sampled_cloud_size'])
                gen_clouds = outputs['p_prior_samples'][-1]

                if kwargs['unit_scale_evaluation']:
                    if kwargs['cloud_scale']:
                        gen_clouds *= kwargs['cloud_scale_scale']
                        ref_clouds *= kwargs['cloud_scale_scale']

                if kwargs['orig_scale_evaluation']:
                    if kwargs['cloud_scale']:
                        gen_clouds *= kwargs['cloud_scale_scale']
                        ref_clouds *= kwargs['cloud_scale_scale']

                    if kwargs['cloud_translate']:
                        shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                        gen_clouds += shift
                        ref_clouds += shift

                    if not kwargs['cloud_rescale2orig']:
                        gen_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                        ref_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    if not kwargs['cloud_recenter2orig']:
                        gen_clouds += batch['orig_c'].unsqueeze(2).cuda()
                        ref_clouds += batch['orig_c'].unsqueeze(2).cuda()

                if kwargs['bbox_scale_evaluation']:
                    if kwargs['cloud_scale']:
                        gen_clouds *= kwargs['cloud_scale_scale']
                        ref_clouds *= kwargs['cloud_scale_scale']

                    if kwargs['cloud_translate']:
                        shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                        gen_clouds += shift
                        ref_clouds += shift

                    if kwargs['cloud_recenter2orig']:
                        gen_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                        ref_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                    if kwargs['cloud_rescale2orig']:
                        gen_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                        ref_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()

                    gen_clouds -= batch['bbox_c'].unsqueeze(2).cuda()
                    ref_clouds -= batch['bbox_c'].unsqueeze(2).cuda()

                    gen_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()
                    ref_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()

                sampled_clouds[i * len(iterator.dataset) + kwargs['batch_size'] * j:
                               i * len(iterator.dataset) + kwargs['batch_size'] * j + gen_clouds.shape[0]] = \
                    gen_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[i * len(iterator.dataset) + kwargs['batch_size'] * j:
                          i * len(iterator.dataset) + kwargs['batch_size'] * j + ref_clouds.shape[0]] = \
                    ref_clouds.cpu().numpy().astype(np.float32)

    if saving_mode:
        clouds_file.close()


def interpolate(iterator, model, **kwargs):
    saving_mode = kwargs.get('saving_mode')
    N_saved_batches = 100

    model.eval()
    torch.set_grad_enabled(False)

    if saving_mode:
        clouds_fname = 'interpolation_{}_{}_{}.h5'.format(kwargs['model_name'][:-4], iterator.dataset.part, kwargs['cloud_size'])
        clouds_fname = kwargs['path2save'] + clouds_fname

        clouds_file = h5.File(clouds_fname, 'w')
        clouds1 = clouds_file.create_dataset(
            'clouds1',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size']),
            dtype=np.float32
        )
        clouds2 = clouds_file.create_dataset(
            'clouds2',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size']),
            dtype=np.float32
        )
        flow_states_clouds1 = clouds_file.create_dataset(
            'flow_states',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size'], 64),
            dtype=np.float32
        )
        interpolations = clouds_file.create_dataset(
            'interpolations',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size'], 9),
            dtype=np.float32
        )

    for i, batch in enumerate(iterator):
        if i == N_saved_batches:
            break

        clouds = batch['cloud'].cuda(non_blocking=True)
        ref_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        inds = np.arange(ref_clouds.shape[0])
        np.random.shuffle(inds)
        ref_clouds = ref_clouds[inds].contiguous()

        codes1 = model.module.encode(clouds)['g_posterior_mus']
        codes2 = model.module.encode(ref_clouds)['g_posterior_mus']

        tmp = model.module.decode(codes1, n_sampled_points=kwargs['cloud_size'])
        fs = torch.cat(list(map(lambda T: T.unsqueeze(3), tmp['p_prior_samples'])), dim=3)

        ints = torch.from_numpy(np.zeros((clouds.shape) + (9,), dtype=np.float32))
        w = torch.from_numpy(np.float32(np.arange(1, 10, 1).reshape(1, 1, -1) / 10)).cuda()
        interpolated_codes = (1. - w) * codes1.unsqueeze(2) + w * codes2.unsqueeze(2)
        for j in range(9):
            outputs = model.module.decode(interpolated_codes[:, :, j], n_sampled_points=kwargs['cloud_size'])
            ints[:, :, :, j] = outputs['p_prior_samples'][-1]

        if saving_mode:
            clouds1[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = clouds.cpu().numpy()
            clouds2[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = ref_clouds.cpu().numpy()
            flow_states_clouds1[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = fs.cpu().numpy()
            interpolations[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = ints.cpu().numpy()

    clouds_file.close()
