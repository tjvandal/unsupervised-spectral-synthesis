'''
Himawari Static
Dropout GOES bands

AHI: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16

1. G16: 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
2. G16: 1,3,4,5,6,7,8,9,10,11,12,13,14,15,16
...
16. G16: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
'''

import os, sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (rank % 4)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"# % (rank % 4)
CUDA_VISIBLE = os.environ["CUDA_VISIBLE_DEVICES"]

import glob
import argparse
import json
import socket

import yaml
import copy
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_net
import torch.multiprocessing as mp
import torch.distributed as dist
import torch

def setup(rank, world_size, port, backend='mpi'):
    host = socket.gethostname()
    devices = torch.cuda.device_count()
    print(f"Rank: {rank}\tWorld size: {world_size}\tPort: {port}\tHost: {host}\tNGPUS: {devices}\tCUDA: {CUDA_VISIBLE}")
    hostname = socket.gethostname()
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(world_rank, args, world_size):
    setup(world_rank, world_size, port=9100)
    device = world_rank % torch.cuda.device_count()

    base_config_file = args.config_file
    base_config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)


    avail_bands = list(range(16))

    to_comma_list = lambda arr: ','.join([str(a) for a in arr])
    comma_to_list = lambda line: [int(i) for i in line.split(",")]

    for i, b in enumerate(avail_bands):
        if i % world_size == world_rank:
            curr_g16_bands = avail_bands[:b] + avail_bands[b+1:]

            model_path = os.path.join(args.experiment_path, f'Band{i}')

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            new_config = copy.deepcopy(base_config)
            new_config['data']['G16']['bands'] = to_comma_list(curr_g16_bands)
            new_config['data']['G16']['dim'] = len(curr_g16_bands)


            shared_names = new_config['data']['G16']['shared'].keys()
            bidx = avail_bands.index(b)
            for sname in shared_names:
                shared_with_s = comma_to_list(new_config['data']['G16']['shared'][sname])
                sharing_from_s = comma_to_list(new_config['data'][sname]['shared']['G16'])

                # b = 0
                if b in shared_with_s:
                    idx = shared_with_s.index(b) # idx = 0
                    shared_with_s[idx:] = [ii-1 for ii in shared_with_s[idx:]] # 
                    #sharing_from_s[idx:] = [ii-1 for ii in sharing_from_s[idx:]]
                    del shared_with_s[idx]
                    del sharing_from_s[idx]
                else:
                    shared_with_s[bidx:] = [ii-1 for ii in shared_with_s[bidx:]]

                new_config['data']['G16']['shared'][sname] = to_comma_list(shared_with_s)
                new_config['data'][sname]['shared']['G16'] = to_comma_list(sharing_from_s)

            new_config['model_path'] = model_path

            if b == 0:
                new_config['gen']['skip_connection'] = 4

            new_config_file = os.path.join(model_path, 'params.yaml')
            with open(new_config_file,'w') as file:
                yaml.dump(new_config, file)

            #print(f"Running Configuration: {json.dumps(new_config, indent=2)}\n")  
            train_net.train_net(new_config, rank=0, device=device, distribute=False)

def run_training(args):
    world_size = int(os.environ['PMI_SIZE'])
    world_rank = int(os.environ['PMI_RANK'])
    main(world_rank, args, world_size)
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default='experiments/1/drop_individual_G16')
    parser.add_argument('--config_file', type=str, default='configs/Base-G16G17H8.yaml')
    args = parser.parse_args()

    run_training(args)
