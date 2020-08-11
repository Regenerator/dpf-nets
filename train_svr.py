import argparse
import os
import io
import yaml

import torch
from torch.utils.data import DataLoader

from lib.datasets.datasets import ShapeNetAllDataset
from lib.datasets.image_transformations import ComposeImageTransformation
from lib.datasets.cloud_transformations import ComposeCloudTransformation

from lib.networks.models import Local_Cond_RNVP_MC_Global_RNVP_VAE_IC
from lib.networks.losses import Local_Cond_RNVP_MC_Global_RNVP_VAE_Loss
from lib.networks.optimizers import Adam, LRUpdater
from lib.networks.training import train


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('config', help='Path to config file in YAML format.')
    parser.add_argument('modelname', help='Model name to save checkpoints.')
    return parser


def save_model(state, model_name):
    torch.save(state, model_name, pickle_protocol=4)
    print('Model saved to ' + model_name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = define_options_parser()
args = parser.parse_args()
with io.open(args.config, 'r') as stream:
    config = yaml.load(stream)
config['model_name'] = '{0}.pkl'.format(args.modelname)
print('Configurations loaded.')

image_transform = ComposeImageTransformation(**config)
cloud_transform = ComposeCloudTransformation(**config)
train_dataset = ShapeNetAllDataset(config['path2data'], part='train',
                                   images_fname=config['images_fname'], meshes_fname=config['meshes_fname'],
                                   cloud_size=config['cloud_size'], return_eval_cloud=True,
                                   return_original_scale=config['cloud_rescale2orig'],
                                   image_transform=image_transform, cloud_transform=cloud_transform,
                                   chosen_label=config['chosen_label'])
print('Dataset init: done.')

train_iterator = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                            num_workers=config['num_workers'], pin_memory=True, drop_last=True)
print('Iterator init: done.')

model = Local_Cond_RNVP_MC_Global_RNVP_VAE_IC(**config).cuda()
print('Model init: done.')
print('Total number of parameters: {}'.format(count_parameters(model)))

criterion = Local_Cond_RNVP_MC_Global_RNVP_VAE_Loss(**config).cuda()
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
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    del(checkpoint)
    print('Model {} loaded.'.format(path2checkpoint))

for epoch in range(cur_epoch, config['n_epochs']):
    train(train_iterator, model, criterion, optimizer, scheduler, epoch, cur_iter, **config)
    cur_iter = 0
