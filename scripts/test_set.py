import os, sys
import glob
import shutil

import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import geonexl1b



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
    
def sensor_files(path, sensor, year, dayofyear, hour=20, minute=0):
    geo = geonexl1b.GeoNEXL1b(path, sensor)
    files = geo.files(year=year, dayofyear=dayofyear)
    if len(files) == 0:
        return None
    files = files[files['hour'] == hour]
    files = files[files['minute'] == minute]
    return files
    
    
def paired_test_set():
    path1 = '/nex/projects/goesscratch/weile/AHI05_java/output/HDF4'
    #G16_path = '/nex/datapool/geonex/public/GOES16/GEONEX-L1B/'
    path2 = '/nex/datapool/geonex/public/GOES17/GEONEX-L1B/'
    sensor1 = 'H8'
    sensor2 = 'G17'

    data_path = 'data/Test/'

    year = 2019
    dayofyear = 2
    hour = 4
    files = test_set_pair(path1, path2, sensor1, sensor2, year, dayofyear, hour=hour)
    for i, row in files.iterrows():
        copy_file(row['file1'], data_path)
        copy_file(row['file2'], data_path)
        
def single_test_set():
    #path = '/nex/datapool/geonex/public/GOES17/GEONEX-L1G/'
    path = '/nex/projects/goesscratch/weile/AHI05_java/output/HDF4'
    sensor = 'H8'
    year = 2019
    dayofyear = 2
    hour = 4
    data_path = '/nobackupp10/tvandal/nex-ai-geo-translation/data/Test/'
    files = sensor_files(path, sensor, year, dayofyear, hour)
    for i, row in files.iterrows():
        print(f"Copying file {row['file']} to {data_path}")
        copy_file(row['file'], data_path)
    
if __name__ == '__main__':
    single_test_set()