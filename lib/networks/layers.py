import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SharedDot(nn.Module):
    def __init__(self, in_features, out_features, n_channels, bias=False,
                 init_weight=None, init_bias=None):
        super(SharedDot, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.weight = nn.Parameter(torch.Tensor(n_channels, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weight:
            nn.init.uniform_(self.weight.data, a=-self.init_weight, b=self.init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight.data, a=0.)
        if self.bias is not None:
            if self.init_bias:
                nn.init.constant_(self.bias.data, self.init_bias)
            else:
                nn.init.constant_(self.bias.data, 0.)

    def forward(self, input):
        output = torch.matmul(self.weight, input.unsqueeze(1))
        if self.bias is not None:
            output.add_(self.bias.unsqueeze(0).unsqueeze(3))
        output.squeeze_(1)
        return output
