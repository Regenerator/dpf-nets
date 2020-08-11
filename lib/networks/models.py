import torch
import torch.nn as nn

from .resnet import resnet18

from .encoders import PointNetCloudEncoder
from .encoders import FeatureEncoder

from .decoders import GlobalRNVPDecoder
from .decoders import LocalCondRNVPDecoder


class Local_Cond_RNVP_MC_Global_RNVP_VAE(nn.Module):
    def __init__(self, **kwargs):
        super(Local_Cond_RNVP_MC_Global_RNVP_VAE, self).__init__()

        self.mode = kwargs.get('usage_mode')
        self.deterministic = kwargs.get('deterministic')

        self.pc_enc_init_n_channels = kwargs.get('pc_enc_init_n_channels')
        self.pc_enc_init_n_features = kwargs.get('pc_enc_init_n_features')
        self.pc_enc_n_features = kwargs.get('pc_enc_n_features')

        self.g_latent_space_size = kwargs.get('g_latent_space_size')

        self.g_prior_n_flows = kwargs.get('g_prior_n_flows')
        self.g_prior_n_features = kwargs.get('g_prior_n_features')

        self.g_posterior_n_layers = kwargs.get('g_posterior_n_layers')

        self.p_latent_space_size = kwargs.get('p_latent_space_size')
        self.p_prior_n_layers = kwargs.get('p_prior_n_layers')

        self.p_decoder_n_flows = kwargs.get('p_decoder_n_flows')
        self.p_decoder_n_features = kwargs.get('p_decoder_n_features')
        self.p_decoder_base_type = kwargs.get('p_decoder_base_type')
        self.p_decoder_base_var = kwargs.get('p_decoder_base_var')

        self.pc_encoder = PointNetCloudEncoder(self.pc_enc_init_n_channels,
                                               self.pc_enc_init_n_features,
                                               self.pc_enc_n_features)

        self.g0_prior_mus = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        self.g0_prior_logvars = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        with torch.no_grad():
            nn.init.normal_(self.g0_prior_mus.data, mean=0.0, std=0.033)
            nn.init.normal_(self.g0_prior_logvars.data, mean=0.0, std=0.33)

        self.g_prior = GlobalRNVPDecoder(self.g_prior_n_flows, self.g_prior_n_features,
                                         self.g_latent_space_size, weight_std=0.01)

        self.g_posterior = FeatureEncoder(self.g_posterior_n_layers, self.pc_enc_n_features[-1],
                                          self.g_latent_space_size, deterministic=False,
                                          mu_weight_std=0.0033, mu_bias=0.0,
                                          logvar_weight_std=0.033, logvar_bias=0.0)

        if self.p_decoder_base_type == 'free':
            self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                          self.p_latent_space_size, deterministic=False,
                                          mu_weight_std=0.001, mu_bias=0.0,
                                          logvar_weight_std=0.01, logvar_bias=0.0)
        elif self.p_decoder_base_type == 'freevar':
            self.register_buffer('p_prior_mus', torch.zeros((1, self.p_latent_space_size, 1)))
            self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                          self.p_latent_space_size, deterministic=True,
                                          mu_weight_std=0.01, mu_bias=0.0)
        elif self.p_decoder_base_type == 'fixed':
            self.register_buffer('p_prior_mus', torch.zeros((1, self.p_latent_space_size, 1)))
            self.register_buffer('p_prior_logvar', self.p_decoder_base_var * torch.ones((1, self.p_latent_space_size, 1)))

        self.pc_decoder = LocalCondRNVPDecoder(self.p_decoder_n_flows,
                                               self.p_decoder_n_features,
                                               self.g_latent_space_size,
                                               weight_std=0.01)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, g_input):
        output = {}

        p_enc_features = self.pc_encoder(g_input)
        g_enc_features = torch.max(p_enc_features, dim=2)[0]

        output['g_posterior_mus'], _ = self.g_posterior(g_enc_features)
        return output

    def decode(self, g_sample, n_sampled_points=2048):
        output = {}

        if self.p_decoder_base_type == 'free':
            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(g_sample)
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]

        elif self.p_decoder_base_type == 'freevar':
            output['p_prior_mus'] = [self.p_prior_mus.expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]
            output['p_prior_logvars'] = [self.p_prior(g_sample).unsqueeze(2).expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]

        elif self.p_decoder_base_type == 'fixed':
            output['p_prior_mus'] = [self.p_prior_mus.expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]
            output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]

        output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]
        buf_p = self.pc_decoder(output['p_prior_samples'][0], g_sample, mode='direct')
        output['p_prior_samples'] += buf_p[0]
        output['p_prior_mus'] += buf_p[1]
        output['p_prior_logvars'] += buf_p[2]
        return output

    def forward(self, g_input, p_input, n_sampled_points=None):
        sampled_cloud_size = p_input.shape[2] if n_sampled_points is None else n_sampled_points

        output = {}
        if self.mode == 'training':
            p_enc_features = self.pc_encoder(g_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]

            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'], output['g_posterior_logvars'])

            output['g_prior_mus'] = [self.g0_prior_mus.expand(g_input.shape[0], self.g_latent_space_size)]
            output['g_prior_logvars'] = [self.g0_prior_logvars.expand(g_input.shape[0], self.g_latent_space_size)]
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            if self.p_decoder_base_type == 'free':
                output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_posterior_samples'])
                output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]
                output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]

            elif self.p_decoder_base_type == 'freevar':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]
                output['p_prior_logvars'] = [self.p_prior(output['g_posterior_samples']).unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]

            elif self.p_decoder_base_type == 'fixed':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]
                output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]

            buf_p = self.pc_decoder(p_input, output['g_posterior_samples'], mode='inverse')
            output['p_prior_samples'] = buf_p[0] + [p_input]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        elif self.mode == 'evaluating':
            p_enc_features = self.pc_encoder(g_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]

            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = output['g_posterior_mus']

            output['g_prior_mus'] = [self.g0_prior_mus.expand(g_input.shape[0], self.g_latent_space_size)]
            output['g_prior_logvars'] = [self.g0_prior_logvars.expand(g_input.shape[0], self.g_latent_space_size)]
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            if self.p_decoder_base_type == 'free':
                output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_posterior_samples'])
                output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'freevar':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior(output['g_posterior_samples']).unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'fixed':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]
            buf = self.pc_decoder(output['p_prior_samples'][0], output['g_posterior_samples'], mode='direct')
            output['p_prior_samples'] += buf[0]
            output['p_prior_mus'] += buf[1]
            output['p_prior_logvars'] += buf[2]

        elif self.mode == 'generating':
            output['g_prior_mus'] = [self.g0_prior_mus.expand(g_input.shape[0], self.g_latent_space_size)]
            output['g_prior_logvars'] = [self.g0_prior_logvars.expand(g_input.shape[0], self.g_latent_space_size)]
            output['g_prior_samples'] = [self.reparameterize(output['g_prior_mus'][0], output['g_prior_logvars'][0])]
            buf_g = self.g_prior(output['g_prior_samples'][0], mode='direct')
            output['g_prior_samples'] += buf_g[0]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            if self.p_decoder_base_type == 'free':
                output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_prior_samples'][-1])
                output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'freevar':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior(output['g_prior_samples'][-1]).unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'fixed':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]

            buf_p = self.pc_decoder(output['p_prior_samples'][0], output['g_prior_samples'][-1], mode='direct')
            output['p_prior_samples'] += buf_p[0]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        return output


class Local_Cond_RNVP_MC_Global_RNVP_VAE_IC(nn.Module):
    def __init__(self, **kwargs):
        super(Local_Cond_RNVP_MC_Global_RNVP_VAE_IC, self).__init__()

        self.mode = kwargs.get('usage_mode')
        self.deterministic = kwargs.get('deterministic')

        self.pc_enc_init_n_channels = kwargs.get('pc_enc_init_n_channels')
        self.pc_enc_init_n_features = kwargs.get('pc_enc_init_n_features')
        self.pc_enc_n_features = kwargs.get('pc_enc_n_features')

        self.g_latent_space_size = kwargs.get('g_latent_space_size')
        self.g_prior_n_layers = kwargs.get('g_prior_n_layers')
        self.g_prior_n_flows = kwargs.get('g_prior_n_flows')
        self.g_prior_n_features = kwargs.get('g_prior_n_features')
        self.g_posterior_n_layers = kwargs.get('g_posterior_n_layers')

        self.p_latent_space_size = kwargs.get('p_latent_space_size')
        self.p_prior_n_layers = kwargs.get('p_prior_n_layers')

        self.p_decoder_base_type = kwargs.get('p_decoder_base_type')
        self.p_decoder_n_flows = kwargs.get('p_decoder_n_flows')
        self.p_decoder_n_features = kwargs.get('p_decoder_n_features')

        self.img_encoder = resnet18(num_classes=self.g_latent_space_size)

        self.pc_encoder = PointNetCloudEncoder(self.pc_enc_init_n_channels,
                                               self.pc_enc_init_n_features,
                                               self.pc_enc_n_features)

        self.g0_prior = FeatureEncoder(self.g_prior_n_layers, self.g_latent_space_size,
                                       self.g_latent_space_size, deterministic=False,
                                       mu_weight_std=0.0033, mu_bias=0.0,
                                       logvar_weight_std=0.033, logvar_bias=0.0)

        self.g_prior = GlobalRNVPDecoder(self.g_prior_n_flows, self.g_prior_n_features,
                                         self.g_latent_space_size, weight_std=0.01)
        self.g_posterior = FeatureEncoder(self.g_posterior_n_layers, self.pc_enc_n_features[-1],
                                          self.g_latent_space_size, deterministic=False,
                                          mu_weight_std=0.0033, mu_bias=0.0,
                                          logvar_weight_std=0.033, logvar_bias=0.0)

        if self.p_decoder_base_type == 'free':
            self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                          self.p_latent_space_size, deterministic=False,
                                          mu_weight_std=0.001, mu_bias=0.0,
                                          logvar_weight_std=0.01, logvar_bias=0.0)
        elif self.p_decoder_base_type == 'freevar':
            self.register_buffer('p_prior_mus', torch.zeros((1, self.p_latent_space_size, 1)))
            self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                          self.p_latent_space_size, deterministic=True,
                                          mu_weight_std=0.01, mu_bias=0.0)
        elif self.p_decoder_base_type == 'fixed':
            self.register_buffer('p_prior_mus', torch.zeros((1, self.p_latent_space_size, 1)))
            self.register_buffer('p_prior_logvar', self.p_decoder_base_var * torch.ones((1, self.p_latent_space_size, 1)))

        self.pc_decoder = LocalCondRNVPDecoder(self.p_decoder_n_flows,
                                               self.p_decoder_n_features,
                                               self.g_latent_space_size,
                                               weight_std=0.01)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, g_input, p_input, images, n_sampled_points=None):
        sampled_cloud_size = p_input.shape[2] if n_sampled_points is None else n_sampled_points

        output = {}
        if self.mode == 'training':
            p_enc_features = self.pc_encoder(g_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]
            img_features = self.img_encoder(images)

            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'], output['g_posterior_logvars'])

            output['g_prior_mus'], output['g_prior_logvars'] = self.g0_prior(img_features)
            output['g_prior_mus'], output['g_prior_logvars'] = [output['g_prior_mus']], [output['g_prior_logvars']]
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            if self.p_decoder_base_type == 'free':
                output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_posterior_samples'])
                output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]
                output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]

            elif self.p_decoder_base_type == 'freevar':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]
                output['p_prior_logvars'] = [self.p_prior(output['g_posterior_samples']).unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]

            elif self.p_decoder_base_type == 'fixed':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]
                output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                    p_input.shape[0], self.p_latent_space_size, p_input.shape[2]
                )]

            buf_p = self.pc_decoder(p_input, output['g_posterior_samples'], mode='inverse')
            output['p_prior_samples'] = buf_p[0] + [p_input]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        elif self.mode == 'evaluating':
            p_enc_features = self.pc_encoder(g_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]
            img_features = self.img_encoder(images)

            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = output['g_posterior_mus']

            output['g_prior_mus'], output['g_prior_logvars'] = self.g0_prior(img_features)
            output['g_prior_mus'], output['g_prior_logvars'] = [output['g_prior_mus']], [output['g_prior_logvars']]
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            if self.p_decoder_base_type == 'free':
                output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_posterior_samples'])
                output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'freevar':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior(output['g_posterior_samples']).unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'fixed':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]

            buf = self.pc_decoder(output['p_prior_samples'][0], output['g_posterior_samples'], mode='direct')
            output['p_prior_samples'] += buf[0]
            output['p_prior_mus'] += buf[1]
            output['p_prior_logvars'] += buf[2]

        elif self.mode == 'predicting':
            img_features = self.img_encoder(images)

            output['g_prior_mus'], output['g_prior_logvars'] = self.g0_prior(img_features)
            output['g_prior_mus'], output['g_prior_logvars'] = [output['g_prior_mus']], [output['g_prior_logvars']]
            output['g_prior_samples'] = [output['g_prior_mus'][0]]
            buf_g = self.g_prior(output['g_prior_samples'][0], mode='direct')
            output['g_prior_samples'] += buf_g[0]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            if self.p_decoder_base_type == 'free':
                output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(output['g_prior_samples'][-1])
                output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'freevar':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior(output['g_prior_samples'][-1]).unsqueeze(2).expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]

            elif self.p_decoder_base_type == 'fixed':
                output['p_prior_mus'] = [self.p_prior_mus.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
                output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                    p_input.shape[0], self.p_latent_space_size, sampled_cloud_size
                )]
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]

            buf_p = self.pc_decoder(output['p_prior_samples'][0], output['g_prior_samples'][-1], mode='direct')
            output['p_prior_samples'] += buf_p[0]
            output['p_prior_mus'] += buf_p[1]
            output['p_prior_logvars'] += buf_p[2]

        return output
