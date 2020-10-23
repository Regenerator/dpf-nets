import argparse
import os
import io
import yaml

import torch
from torch.utils.data import DataLoader

from lib.datasets.datasets import ShapeNetCoreDataset
from lib.datasets.cloud_transformations import ComposeCloudTransformation

from lib.networks.models import Local_Cond_RNVP_MC_Global_RNVP_VAE
from lib.networks.losses import Local_Cond_RNVP_MC_Global_RNVP_VAE_Loss
from lib.networks.optimizers import Adam, LRUpdater
from lib.networks.training import train
from lib.networks.utils import cnt_params


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('config', type=str, help='Path to config file in YAML format.')
    parser.add_argument('modelname', type=str, help='Model name to save checkpoints.')
    parser.add_argument('n_epochs', type=int, help='Total number of training epochs.')
    parser.add_argument('lr', type=float, help='Learining rate value.')
    parser.add_argument('--resume', action='store_true',
                        help='Flag signaling if training is resumed from a checkpoint.')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Flag signaling if optimizer parameters are resumed from a checkpoint.')
    return parser


parser = define_options_parser()
args = parser.parse_args()
with io.open(args.config, 'r') as stream:
    config = yaml.load(stream)
config['model_name'] = '{0}.pkl'.format(args.modelname)
config['n_epochs'] = args.n_epochs
config['min_lr'] = config['max_lr'] = args.lr
config['resume'] = True if args.resume else False
config['resume_optimizer'] = True if args.resume_optimizer else False
print('Configurations loaded.')

cloud_transform = ComposeCloudTransformation(**config)
train_dataset = ShapeNetCoreDataset(config['path2data'],
                                    part='train', meshes_fname=config['meshes_fname'],
                                    cloud_size=config['cloud_size'], return_eval_cloud=True,
                                    return_original_scale=config['cloud_rescale2orig'],
                                    cloud_transform=cloud_transform,
                                    chosen_label=config['chosen_label'])
print('Dataset init: done.')

train_iterator = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                            num_workers=config['num_workers'], pin_memory=True, drop_last=True)
print('Iterator init: done.')

model = Local_Cond_RNVP_MC_Global_RNVP_VAE(**config).cuda()
print('Model init: done.')
print('Total number of parameters: {}'.format(cnt_params(model.parameters())))

criterion = Local_Cond_RNVP_MC_Global_RNVP_VAE_Loss(**config).cuda()
print('Loss init: done.')

optimizer = Adam(model.parameters(), lr=config['max_lr'], weight_decay=config['wd'],
                 betas=(config['beta1'], config['max_beta2']), amsgrad=True)
scheduler = LRUpdater(len(train_iterator), **config)
print('Optimizer init: done.')

if not config['resume']:
    cur_epoch = 0
    cur_iter = 0
else:
    path2checkpoint = os.path.join(config['path2save'], 'models', 'DPFNets', config['model_name'])
    checkpoint = torch.load(path2checkpoint)
    cur_epoch = checkpoint['epoch']
    cur_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['model_state'])
    if config['resume_optimizer']:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    del(checkpoint)
    print('Model {} loaded.'.format(path2checkpoint))

for epoch in range(cur_epoch, config['n_epochs']):
    train(train_iterator, model, criterion, optimizer, scheduler, epoch, cur_iter, **config)
    cur_iter = 0
