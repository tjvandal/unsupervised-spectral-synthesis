import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
import geonexl1b

import torch
from torch.utils import data

import pandas as pd
import numpy as np


#directory = '/nobackupp10/nexprojects/GEONEX/datapool/HM08_AHI_v2/HM08_AHI05/'
#sensor = 'AHI'
#year = 2016

directory = '/nex/datapool/geonex/public/GOES16/GEONEX-L1B/'
sensor = 'ABI'
year = 2019

geo = geonexl1b.GeoNEXL1b(data_directory=directory, sensor=sensor)
dfs = []
for d in range(2, 360, 15):
    dfs.append(geo.files(tile=None, year=year, dayofyear=d))
df = pd.concat(dfs)
files = df.file.values

np.random.shuffle(files)

dataset = dataloader.GeoNEXData(files[:1000], sensor, patch_size=500)
data_params = {'batch_size': 16, 'shuffle': True,
               'num_workers': 8, 'pin_memory': False}
generator = data.DataLoader(dataset, **data_params)

mu, sd = [], []
for x in generator:
    mu.append(np.mean(x.numpy(), (0,2,3)))
    sd.append(np.std(x.numpy(), (0,2,3)))
print(sd)
mu = sum(mu) / len(mu)
sd = sum(sd) / len(sd)
print(','.join(['%2.2f' % m for m in mu]))
print(','.join(['%2.2f' % m for m in sd]))
