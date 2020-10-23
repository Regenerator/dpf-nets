import math
import numpy as np
import torch

from torch.optim import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt()
                else:
                    denom = exp_avg_sq.sqrt()

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = math.sqrt(1 - beta2 ** state['step'])

                exp_avg_c = exp_avg / bias_correction1
                denom_c = torch.add(denom / bias_correction2, group['eps'])

                if group['weight_decay'] != 0:
                    p.data.add_(-torch.addcdiv(
                        torch.mul(p.data, group['weight_decay']), exp_avg_c, denom_c, value=group['lr']
                    ))
                else:
                    p.data.addcdiv_(exp_avg_c, denom_c, value=-group['lr'])

        return loss


class LRUpdater(object):
    def __init__(self, epoch_length, **kwargs):
        self.epoch_length = epoch_length
        self.cycle_length = kwargs['cycle_length']
        self.min_lr = kwargs['min_lr']
        self.max_lr = kwargs['max_lr']
        self.beta1 = kwargs['beta1']
        self.min_beta2 = kwargs['min_beta2']
        self.max_beta2 = kwargs['max_beta2']

    def __call__(self, optimizer, epoch, iteration):
        rel_epoch = (epoch) % self.cycle_length
        cur_step = (rel_epoch * self.epoch_length + iteration) / (self.cycle_length * self.epoch_length)
        cur_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(np.pi * cur_step))
        cur_beta2 = self.min_beta2 + 0.5 * (self.max_beta2 - self.min_beta2) * (1.0 + np.cos(np.pi * cur_step))

        for group in optimizer.param_groups:
            group['lr'] = cur_lr
            group['betas'] = (self.beta1, cur_beta2)
