import torch.nn as nn

from .flows import RealNVPFlowCouple
from .flows import CondRealNVPFlow3DTriple


class GlobalRNVPDecoder(nn.Module):
    def __init__(self, n_flows, n_features, g_n_features, weight_std=0.01):
        super(GlobalRNVPDecoder, self).__init__()
        self.n_flows = n_flows
        self.n_features = n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std

        self.flows = nn.ModuleList([
            RealNVPFlowCouple(n_features, g_n_features, weight_std=self.weight_std, pattern=(i % 2))
            for i in range(n_flows)
        ])

    def forward(self, g, mode='direct'):
        gs = []
        mus = []
        logvars = []
        for i in range(self.n_flows):
            if mode == 'direct':
                cur_g = g if i == 0 else gs[-1]
                buf = self.flows[i](cur_g, mode=mode)
                gs = gs + buf[0]
                mus = mus + buf[1]
                logvars = logvars + buf[2]
            elif mode == 'inverse':
                cur_g = g if i == 0 else gs[0]
                buf = self.flows[-(i + 1)](cur_g, mode=mode)
                gs = buf[0] + gs
                mus = buf[1] + mus
                logvars = buf[2] + logvars

        return gs, mus, logvars


class LocalCondRNVPDecoder(nn.Module):
    def __init__(self, n_flows, f_n_features, g_n_features, weight_std=0.01):
        super(LocalCondRNVPDecoder, self).__init__()
        self.n_flows = n_flows
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std

        self.flows = nn.ModuleList(
            [CondRealNVPFlow3DTriple(f_n_features, g_n_features,
                                     weight_std=self.weight_std, pattern=(i % 2)) for i in range(n_flows)]
        )

    def forward(self, p, g, mode='direct'):
        ps = []
        mus = []
        logvars = []
        for i in range(self.n_flows):
            if mode == 'direct':
                cur_p = p if i == 0 else ps[-1]
                buf = self.flows[i](cur_p, g, mode=mode)
                ps = ps + buf[0]
                mus = mus + buf[1]
                logvars = logvars + buf[2]
            elif mode == 'inverse':
                cur_p = p if i == 0 else ps[0]
                buf = self.flows[-(i + 1)](cur_p, g, mode=mode)
                ps = buf[0] + ps
                mus = buf[1] + mus
                logvars = buf[2] + logvars

        return ps, mus, logvars
