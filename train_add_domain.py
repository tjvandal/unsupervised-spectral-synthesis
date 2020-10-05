import os, sys
import copy
import glob
import yaml
import argparse

import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
torch.cuda.empty_cache()

import numpy as np
import pandas as pd

from data import geonexl1b
from trainer import Trainer
import utils

from data import petastorm_reader

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train_net_mp(rank, world_size, port, params, new_model_path):
    setup(rank, world_size, port)
    train_net(params, new_model_path, rank=rank)

def train_net(params, new_model_path, rank=None, device=None, distribute=True):
    if rank is None:
        rank = 0
        setup(rank, 1, 9100+rank)

    print(f"Running training on rank {rank}.")
    # initialize trainer
    if device is None:
        device = rank % torch.cuda.device_count()

    trainer = Trainer(params, distribute=distribute, rank=rank, gpu=device)

    # set device
    if rank == 0:
        trainer.load_checkpoint()


    data_params = copy.deepcopy(params)
    params['data']['H8_15'] = {'data_url': params['data']['H8']['data_url'],
                            'bands': '0,2,3,4,5,6,7,8,9,10,11,12,13,14,15',
                            'shared': {},
                            'dim': 15}
    params['data']['H8_15']['shared'] = {'H8': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14',
                                         'G16': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14',
                                         'G17': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14'}
    params['data']['H8']['shared']['H8_15'] = '0,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
    params['data']['G16']['shared']['H8_15'] = params['data']['G16']['shared']['H8']
    params['data']['G17']['shared']['H8_15'] = params['data']['G17']['shared']['H8']
    trainer.params = params

    # add satellite to domain
    trainer.add_domain('H8_15', 15)
    trainer.dis = trainer.dis.to(device)
    trainer.gen = trainer.gen.to(device)
    trainer.checkpoint_filepath = os.path.join(new_model_path, 'checkpoint.flownet.pth.tar')
    trainer.global_step = 0

    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path)

    new_params_file = yaml.dump(params, open(os.path.join(new_model_path, 'params.yaml'), 'w'))

    #if rank == 0:
    trainer.tfwriter = SummaryWriter(os.path.join(new_model_path, 'tfsummary'))

    for name, p in trainer.gen.named_parameters():
        if 'H8_15' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


    # Load optimizer
    trainer.optimizer_gen = torch.optim.Adam([p for p in trainer.gen.parameters() if p.requires_grad],
                                              lr=params['lr'],
                                              betas=(params['beta1'], params['beta2']),
                                              weight_decay=params['weight_decay'])
    trainer.optimizer_dis = torch.optim.Adam([p for p in trainer.dis.parameters() if p.requires_grad],
                                              lr=params['lr'],
                                              betas=(params['beta1'], params['beta2']),
                                              weight_decay=params['weight_decay'])


    # Load dataset
    data_params = copy.deepcopy(params)
    del data_params['data']['G16']
    del data_params['data']['G17']
    #del data_params['data']['H8']
    data_generator = petastorm_reader.make_L1G_generators(data_params)


    while trainer.global_step < params['max_iter']:
        try:
            for batch_idx, sample in enumerate(data_generator):
                #for batch_idx, x_list in enumerate(loader):
                #x_dict = {n: x.to(device) for n, x in zip(datanames, x_list)}
                x_dict = {n: x.to(device) for n, x in sample.items()}
                log = False
                if (trainer.global_step % params['log_iter'] == 0) and (rank == 0):
                    log = True
                loss_gen = trainer.gen_update(x_dict, log=log)
                loss_dis = trainer.dis_update(x_dict, log=log)

                if rank == 0:
                    if log:
                        print(f"Step {trainer.global_step} -- Generator={loss_gen.item():4.4g}, Discriminator={loss_dis.item():4.4g}")
                        #print(f"Step {trainer.global_step} -- Generator={loss_gen.item():4.4g}")

                    trainer.update_step()
                    if trainer.global_step % params['checkpoint_step'] == 1:
                        trainer.save_checkpoint()

                    if trainer.global_step >= params['max_iter']:
                        break
        finally:
            #loaders = petastorm_reader.make_loaders(params)
            data_generator = petastorm_reader.make_L1G_generators(params)

    if rank == 0:
        trainer.save_checkpoint()

def run_training(params,new_model_path, world_size, port):
    params['batch_size'] = params['batch_size'] // world_size
    mp.spawn(train_net_mp,
             args=(world_size, port, params, new_model_path),
             nprocs=world_size,
             join=True)
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--port', type=int, default=9001)

    args = parser.parse_args()

    
    pretrained_config = '/nobackupp10/tvandal/nex-ai-geo-translation/configs/Base-G16G17H8.yaml'
    new_model_path = '/nobackupp10/tvandal/nex-ai-geo-translation/experiments/1/G16G17H8_Add_H8_15_v0.2/'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    params = utils.get_config(pretrained_config)
    
    run_training(params, new_model_path, args.world_size, args.port)

if __name__ == "__main__":
    main()
