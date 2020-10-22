import os
from time import time
from sys import stdout

import h5py as h5
import numpy as np
import torch

from lib.networks.utils import AverageMeter
from lib_networks.utils import distChamferCUDA, f_score


def evaluate(iterator, model, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    util_mode = kwargs.get('util_mode')
    is_saving = kwargs.get('saving')

    if is_saving:
        clouds_fname = '{}_{}_{}_{}_clouds_{}.h5'.format(kwargs['model_name'][:-4],
                                                         iterator.dataset.part,
                                                         kwargs['cloud_size'],
                                                         kwargs['sampled_cloud_size'],
                                                         util_mode)
        clouds_fname = os.path.join(kwargs['path2save'], clouds_fname)
        clouds_file = h5.File(clouds_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(kwargs['N_sets'] * len(iterator.dataset), 3, kwargs['sampled_cloud_size']),
            dtype=np.float32
        )
        gt_clouds = clouds_file.create_dataset(
            'gt_clouds',
            shape=(kwargs['N_sets'] * len(iterator.dataset), 3, kwargs['cloud_size']),
            dtype=np.float32
        )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    inf_time = AverageMeter()

    if util_mode == 'training':
        LB = AverageMeter()
        PNLL = AverageMeter()
        GNLL = AverageMeter()
        GENT = AverageMeter()

    elif util_mode == 'evaluating':
        CD = AverageMeter()

    elif util_mode == 'generating':
        gen_clouds_buf = []
        ref_clouds_buf = []

    elif util_mode == 'predicting':
        CD = AverageMeter()
        F1 = AverageMeter()

    model.eval()
    torch.set_grad_enabled(False)

    end = time()
    for i, batch in enumerate(iterator):
        data_time.update(time() - end)

        g_clouds = batch['cloud'].cuda(non_blocking=True)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)

        if train_mode == 'p_rnvp_mc_g_rnvp_vae':
            inf_end = time()
            outputs = model(g_clouds, p_clouds, n_sampled_points=kwargs['sampled_cloud_size'])
        elif train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
            images = batch['image'].cuda(non_blocking=True)
            inf_end = time()
            outputs = model(g_clouds, p_clouds, images, n_sampled_points=kwargs['sampled_cloud_size'])
        inf_time.update((time() - inf_end) / g_clouds.shape[0], g_clouds.shape[0])

        if util_mode == 'training':
            loss, pnll, gnll, gent = loss_func(g_clouds, p_clouds, outputs)
            LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])
            PNLL.update(pnll.item(), clouds.shape[0])
            GNLL.update(gnll.item(), clouds.shape[0])
            GENT.update(gent.item(), clouds.shape[0])

        elif util_mode == 'evaluating':
            r_clouds = outputs['p_prior_samples'][-1]

            if kwargs['orig_scale_evaluation']:
                if kwargs['cloud_scale']:
                    r_clouds *= kwargs['cloud_scale_scale']
                    p_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    r_clouds += shift
                    p_clouds += shift

                if not kwargs['cloud_rescale2orig']:
                    r_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    p_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                if not kwargs['cloud_recenter2orig']:
                    r_clouds += batch['orig_c'].unsqueeze(2).cuda()
                    p_clouds += batch['orig_c'].unsqueeze(2).cuda()

            if is_saving:
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + r_clouds.shape[0]] = \
                    r_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + p_clouds.shape[0]] = \
                    p_clouds.cpu().numpy().astype(np.float32)

            dl, dr = distChamferCUDA(torch.transpose(r_clouds, 1, 2).contiguous(),
                                     torch.transpose(p_clouds, 1, 2).contiguous())
            cd = (dl.mean(1) + dr.mean(1)).mean()
            CD.update(cd.item(), p_clouds.shape[0])

        elif util_mode == 'generating':
            r_clouds = outputs['p_prior_samples'][-1]

            if kwargs['orig_scale_evaluation']:
                if kwargs['cloud_scale']:
                    r_clouds *= kwargs['cloud_scale_scale']
                    p_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    r_clouds += shift
                    p_clouds += shift

                if not kwargs['cloud_rescale2orig']:
                    r_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    p_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                if not kwargs['cloud_recenter2orig']:
                    r_clouds += batch['orig_c'].unsqueeze(2).cuda()
                    p_clouds += batch['orig_c'].unsqueeze(2).cuda()

            gen_clouds_buf.append(r_clouds)
            ref_clouds_buf.append(p_clouds)

            if is_saving:
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + r_clouds.shape[0]] = \
                    r_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + p_clouds.shape[0]] = \
                    p_clouds.cpu().numpy().astype(np.float32)

        elif util_mode == 'predicting':
            r_clouds = outputs['p_prior_samples'][-1]

            if kwargs['unit_scale_evaluation']:
                if kwargs['cloud_scale']:
                    r_clouds *= kwargs['cloud_scale_scale']
                    p_clouds *= kwargs['cloud_scale_scale']

            if kwargs['orig_scale_evaluation']:
                if kwargs['cloud_scale']:
                    r_clouds *= kwargs['cloud_scale_scale']
                    p_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    r_clouds += shift
                    p_clouds += shift

                if not kwargs['cloud_rescale2orig']:
                    r_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    p_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                if not kwargs['cloud_recenter2orig']:
                    r_clouds += batch['orig_c'].unsqueeze(2).cuda()
                    p_clouds += batch['orig_c'].unsqueeze(2).cuda()

            if kwargs['bbox_scale_evaluation']:
                if kwargs['cloud_scale']:
                    r_clouds *= kwargs['cloud_scale_scale']
                    p_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    r_clouds += shift
                    p_clouds += shift

                if kwargs['cloud_recenter2orig']:
                    r_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                    p_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                if kwargs['cloud_rescale2orig']:
                    r_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    p_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()

                r_clouds -= batch['bbox_c'].unsqueeze(2).cuda()
                p_clouds -= batch['bbox_c'].unsqueeze(2).cuda()

                r_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()
                p_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()

            if is_saving:
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + r_clouds.shape[0]] = \
                    r_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + p_clouds.shape[0]] = \
                    p_clouds.cpu().numpy().astype(np.float32)

            r_clouds = torch.transpose(r_clouds, 1, 2).contiguous()
            p_clouds = torch.transpose(p_clouds, 1, 2).contiguous()

            dl, dr = distChamferCUDA(r_clouds, p_clouds)
            cd = (dl.mean(1) + dr.mean(1)).mean()
            f1 = f_score(r_clouds, p_clouds).mean()
            CD.update(cd.item(), p_clouds.shape[0])
            F1.update(f1.item(), p_clouds.shape[0])

        batch_time.update(time() - end)
        line = '[{cc}/{il}]  \tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(cc=i + 1, il=len(iterator), batch_time=batch_time)
        if util_mode == 'training':
            line += '\tLB {LB.val:.2f} ({LB.avg:.2f})'.format(LB=LB)
            line += '\tPNLL {PNLL.val:.2f} ({PNLL.avg:.2f})'.format(PNLL=PNLL)
            line += '\tGNLL {GNLL.val:.2f} ({GNLL.avg:.2f})'.format(GNLL=GNLL)
            line += '\tGENT {GENT.val:.2f} ({GENT.avg:.2f})'.format(GENT=GENT)
        elif util_mode == 'evaluating':
            line += '\tCD {CD.val:.6f} ({CD.avg:.6f})'.format(CD=CD)
        elif util_mode == 'predicting':
            line += '\tCD {CD.val:.6f} ({CD.avg:.6f})'.format(CD=CD)
            line += '\tF1 {F1.val:.1f} ({F1.avg:.1f})'.format(F1=F1)
        line += '\n'
        stdout.write(line)
        stdout.flush()
        end = time()

    print('Inference time: {} sec/sample'.format(inf_time.avg))

    if util_mode == 'evaluating':
        print('CD: {:.6f}'.format(CD.avg))

    elif util_mode == 'predicting':
        print('CD: {:.6f}'.format(CD.avg))
        print('F1: {:.1f}'.format(F1.avg))

    elif util_mode == 'generating':
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

    if is_saving and kwargs['N_sets'] > 1:
        for i in range(1, kwargs['N_sets'], 1):
            for j, batch in enumerate(iterator):
                g_clouds = batch['cloud'].cuda(non_blocking=True)
                p_clouds = batch['eval_cloud'].cuda(non_blocking=True)
                outputs = model(g_clouds, p_clouds, n_sampled_points=kwargs['sampled_cloud_size'])
                r_clouds = outputs['p_prior_samples'][-1]

                if kwargs['unit_scale_evaluation']:
                    if kwargs['cloud_scale']:
                        r_clouds *= kwargs['cloud_scale_scale']
                        p_clouds *= kwargs['cloud_scale_scale']

                if kwargs['orig_scale_evaluation']:
                    if kwargs['cloud_scale']:
                        r_clouds *= kwargs['cloud_scale_scale']
                        p_clouds *= kwargs['cloud_scale_scale']

                    if kwargs['cloud_translate']:
                        shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                        r_clouds += shift
                        p_clouds += shift

                    if not kwargs['cloud_rescale2orig']:
                        r_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                        p_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    if not kwargs['cloud_recenter2orig']:
                        r_clouds += batch['orig_c'].unsqueeze(2).cuda()
                        p_clouds += batch['orig_c'].unsqueeze(2).cuda()

                if kwargs['bbox_scale_evaluation']:
                    if kwargs['cloud_scale']:
                        r_clouds *= kwargs['cloud_scale_scale']
                        p_clouds *= kwargs['cloud_scale_scale']

                    if kwargs['cloud_translate']:
                        shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                        r_clouds += shift
                        p_clouds += shift

                    if kwargs['cloud_recenter2orig']:
                        r_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                        p_clouds -= batch['orig_c'].unsqueeze(2).cuda()
                    if kwargs['cloud_rescale2orig']:
                        r_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                        p_clouds /= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()

                    r_clouds -= batch['bbox_c'].unsqueeze(2).cuda()
                    p_clouds -= batch['bbox_c'].unsqueeze(2).cuda()

                    r_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()
                    p_clouds /= batch['bbox_s'].unsqueeze(1).unsqueeze(2).cuda()

                sampled_clouds[i * len(iterator.dataset) + kwargs['batch_size'] * j:
                               i * len(iterator.dataset) + kwargs['batch_size'] * j + r_clouds.shape[0]] = \
                    r_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[i * len(iterator.dataset) + kwargs['batch_size'] * j:
                          i * len(iterator.dataset) + kwargs['batch_size'] * j + p_clouds.shape[0]] = \
                    p_clouds.cpu().numpy().astype(np.float32)

    if is_saving:
        clouds_file.close()


# def interpolate(iterator, model, **kwargs):
#     saving_mode = kwargs.get('saving_mode')
#     N_saved_batches = 100

#     model.eval()
#     torch.set_grad_enabled(False)

#     if saving_mode:
#         clouds_fname = 'interpolation_{}_{}_{}.h5'.format(kwargs['model_name'][:-4], iterator.dataset.part, kwargs['cloud_size'])
#         clouds_fname = kwargs['path2save'] + clouds_fname

#         clouds_file = h5.File(clouds_fname, 'w')
#         clouds1 = clouds_file.create_dataset(
#             'clouds1',
#             shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size']),
#             dtype=np.float32
#         )
#         clouds2 = clouds_file.create_dataset(
#             'clouds2',
#             shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size']),
#             dtype=np.float32
#         )
#         flow_states_clouds1 = clouds_file.create_dataset(
#             'flow_states',
#             shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size'], 64),
#             dtype=np.float32
#         )
#         interpolations = clouds_file.create_dataset(
#             'interpolations',
#             shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size'], 9),
#             dtype=np.float32
#         )

#     for i, batch in enumerate(iterator):
#         if i == N_saved_batches:
#             break

#         clouds = batch['cloud'].cuda(non_blocking=True)
#         ref_clouds = batch['eval_cloud'].cuda(non_blocking=True)
#         inds = np.arange(ref_clouds.shape[0])
#         np.random.shuffle(inds)
#         ref_clouds = ref_clouds[inds].contiguous()

#         codes1 = model.module.encode(clouds)['g_posterior_mus']
#         codes2 = model.module.encode(ref_clouds)['g_posterior_mus']

#         tmp = model.module.decode(codes1, n_sampled_points=kwargs['cloud_size'])
#         fs = torch.cat(list(map(lambda T: T.unsqueeze(3), tmp['p_prior_samples'])), dim=3)

#         ints = torch.from_numpy(np.zeros((clouds.shape) + (9,), dtype=np.float32))
#         w = torch.from_numpy(np.float32(np.arange(1, 10, 1).reshape(1, 1, -1) / 10)).cuda()
#         interpolated_codes = (1. - w) * codes1.unsqueeze(2) + w * codes2.unsqueeze(2)
#         for j in range(9):
#             outputs = model.module.decode(interpolated_codes[:, :, j], n_sampled_points=kwargs['cloud_size'])
#             ints[:, :, :, j] = outputs['p_prior_samples'][-1]

#         if saving_mode:
#             clouds1[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = clouds.cpu().numpy()
#             clouds2[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = ref_clouds.cpu().numpy()
#             flow_states_clouds1[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = fs.cpu().numpy()
#             interpolations[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = ints.cpu().numpy()

#     clouds_file.close()
