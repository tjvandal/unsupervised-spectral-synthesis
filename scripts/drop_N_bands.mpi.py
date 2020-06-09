'''
Experiment to test how many bands are needed for reconstruction

Creates processes for removing more and more bands in order 
    8, 3, 12, 1, 14, 11, 5, 2, 9, 7, 13, 4, 10, 6, 15
    
Dropout G16 bands
G17/H8 Static
'''

import os, sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (rank % 4)

import glob
import argparse

import yaml
import copy
import numpy as np

import torch.distributed as dist
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_net


def setup(rank, world_size, port, backend='mpi'):
    host = socket.gethostname()
    print(f"Rank: {rank}    World size: {world_size}   Port: {port}, Host: {host}")
    
    hostname = socket.gethostname()
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(world_rank, world_size, args):
    base_config = yaml.load(open(args.base_config_file), Loader=yaml.FullLoader)
    avail_bands = list(range(16))
    drop_g16_bands = [8, 3, 12, 1, 14, 11, 5, 2, 9, 7, 13, 4, 10, 6, 15]
    curr_bands = avail_bands

    experiment_bands = [curr_bands]
    for b in drop_g16_bands:
        i = curr_bands.index(b)
        curr_bands = curr_bands[:i] + curr_bands[i+1:]
        experiment_bands.append(curr_bands)

    print(experiment_bands)

    to_comma_list = lambda arr: ','.join([str(a) for a in arr])
    comma_to_list = lambda line: [int(i) for i in line.split(",")]

    for i, bands in enumerate(experiment_bands[::-1]):
        if i % world_size == world_rank:
            N_bands = len(bands)

            model_path = os.path.join(args.experiment_path, f'N_{N_bands:02g}')

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            new_config = copy.deepcopy(base_config)
            new_config['data']['G16']['bands'] = to_comma_list(bands)
            new_config['data']['G16']['dim'] = N_bands

            shared_names = new_config['data']['G16']['shared'].keys() 

            for dname in shared_names:
                sharing_all = comma_to_list(new_config['data']['G16']['shared'][dname])
                sharing_curr = np.intersect1d(sharing_all, bands)
                sharing_idxs = [np.argwhere(bands == v)[0,0] for v in sharing_curr]
                new_config['data']['G16']['shared'][dname] = to_comma_list(sharing_idxs)

                sharing_from_d = comma_to_list(new_config['data'][dname]['shared']['G16'])
                idxs_sharing_from_d = [np.argwhere(sharing_all == v)[0,0] for v in sharing_curr]

                new_config['data'][dname]['shared']['G16'] = to_comma_list(np.asarray(sharing_from_d)[idxs_sharing_from_d])
                #print(sharing_idxs, sharing_curr[sharing_idxs])

            new_config['model_path'] = model_path

            new_config_file = os.path.join(model_path, 'params.yaml')
            with open(new_config_file,'w') as file:
                yaml.dump(new_config, file)

            print(f"Running Configuration: {new_config}")

            #train_net.train_net(new_config)
            device = world_rank % torch.cuda.device_count()
            train_net_multigpu.train_net(new_config, rank=0, device=device, distribute=False)

def run_training(args):
    world_size = int(os.environ['PMI_SIZE'])
    world_rank = int(os.environ['PMI_RANK'])
    main(world_rank, world_size, args)
    #cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default='.tmp/models/drop_GOES_bands_v2.1/')
    parser.add_argument('--base_config_file', type=str, default='configs/Base-G16G17H8.yaml')
    args = parser.parse_args()

    run_training(args)
