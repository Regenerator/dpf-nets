from collections import OrderedDict

import torch
import torch.nn as nn

from .layers import SharedDot, Swish


class PointNetCloudEncoder(nn.Module):
    def __init__(self, init_n_channels, init_n_features, n_features):
        super(PointNetCloudEncoder, self).__init__()
        self.init_n_channels = init_n_channels
        self.init_n_features = init_n_features
        self.n_features = n_features

        self.features = nn.Sequential(OrderedDict([
            ('init_sd', SharedDot(init_n_channels, init_n_features, 1, bias=False)),
            ('init_sd_bn', nn.BatchNorm1d(init_n_features)),
            ('init_sd_relu', nn.ReLU(inplace=True))
        ]))
        for i in range(len(self.n_features)):
            cur_features = init_n_features if i == 0 else n_features[i - 1]
            self.features.add_module('sd{}'.format(i), SharedDot(cur_features, n_features[i], 1, bias=False))
            self.features.add_module('sd{}_bn'.format(i), nn.BatchNorm1d(n_features[i]))
            self.features.add_module('sd{}_relu'.format(i), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.features(input)


class FeatureEncoder(nn.Module):
    def __init__(self, n_layers, in_features, latent_space_size,
                 deterministic=False, batch_norm=True,
                 mu_weight_std=0.001, mu_bias=0.0,
                 logvar_weight_std=0.01, logvar_bias=0.0,
                 easy_init=False):
        super(FeatureEncoder, self).__init__()
        self.n_layers = n_layers
        self.in_features = in_features
        self.latent_space_size = latent_space_size
        self.deterministic = deterministic
        self.batch_norm = batch_norm
        self.mu_weight_std = mu_weight_std
        self.mu_bias = mu_bias
        self.logvar_weight_std = logvar_weight_std
        self.logvar_bias = logvar_bias
        self.easy_init = easy_init

        if n_layers > 0:
            self.features = nn.Sequential()
            for i in range(n_layers):
                self.features.add_module('mlp{}'.format(i), nn.Linear(in_features, in_features, bias=False))
                if self.batch_norm:
                    self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(in_features))
                self.features.add_module('mlp{}_swish'.format(i), Swish())

        self.mus = nn.Sequential(OrderedDict([
            ('mu_mlp0', nn.Linear(in_features, latent_space_size, bias=True))
        ]))
        if not easy_init:
            with torch.no_grad():
                self.mus[-1].weight.data.normal_(std=mu_weight_std)
                nn.init.constant_(self.mus[-1].bias.data, self.mu_bias)

        if not self.deterministic:
            self.logvars = nn.Sequential(OrderedDict([
                ('logvar_mlp0', nn.Linear(in_features, latent_space_size, bias=True))
            ]))
            if not easy_init:
                with torch.no_grad():
                    self.logvars[-1].weight.data.normal_(std=logvar_weight_std)
                    nn.init.constant_(self.logvars[-1].bias.data, self.logvar_bias)

    def forward(self, input):
        if self.n_layers > 0:
            features = self.features(input)
        else:
            features = input

        if self.deterministic:
            return self.mus(features)
        else:
            return self.mus(features), self.logvars(features)
