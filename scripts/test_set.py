import os, sys
import glob
import shutil

import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geonexl1b

path1 = '/nex/projects/goesscratch/weile/AHI05_java/output/HDF4'
#G16_path = '/nex/datapool/geonex/public/GOES16/GEONEX-L1B/'
path2 = '/nex/datapool/geonex/public/GOES17/GEONEX-L1B/'
sensor1 = 'H8'
sensor2 = 'G17'

data_path = 'data/Test/'

year = 2019
dayofyear = 2
hour = 4

def test_set_pair(path1, path2, sensor1, sensor2, year, dayofyear,
                 hour=20, minute=0):
    pair = geonexl1b.L1bPaired(path1, path2, sensor1, sensor2)
    files = pair.files(year=year, dayofyear=dayofyear, how='outer')
    files = files[files['hour1'] == hour]
    files = files[files['minute1'] == minute]
    return files

def copy_file(file, dest):
    if not isinstance(file, str):
        print(f"File is not string: {file}")
        return False
    src_sub = '/'.join(file.split('/')[-6:])
    dest = os.path.join(dest, src_sub)
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    shutil.copy(file, dest)


files = test_set_pair(path1, path2, sensor1, sensor2, year, dayofyear, hour=hour)
for i, row in files.iterrows():
    copy_file(row['file1'], data_path)
    copy_file(row['file2'], data_path)