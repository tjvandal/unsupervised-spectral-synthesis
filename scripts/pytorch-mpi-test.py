import os
import socket
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size, hostname):
    print(f"I am {rank} of {size} in {hostname}")


def init_processes(rank, size, hostname, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, hostname)
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = int(os.environ['PMI_SIZE'])
    world_rank = int(os.environ['PMI_RANK'])
    hostname = socket.gethostname()
    
    init_processes(world_rank, world_size, hostname, run, backend='mpi')