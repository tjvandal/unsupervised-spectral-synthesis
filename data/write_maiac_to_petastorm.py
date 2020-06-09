import os, sys
import glob
import argparse

import torch
import pyhdf

import geonexl1b
import maiac_data
from schema import MAIACSchema, MAIACSchema256

import numpy as np
import xarray as xr

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row

def file_examples(ahi05file, ahi12file, patch_size):
    try:
        x = geonexl1b.L1bFile(ahi05file, resolution_km=2.).load()
        y = maiac_data.AHI12(ahi12file, resolution_km=2.).load()
        x[x == 0] = np.nan
        xmin = np.nanmin(x, axis=(0,1))
        xmax = np.nanmax(x, axis=(0,1))
        if xmin[11] < 100:
            print('low value for band 12 in', filepath)
            sys.exit()
    except pyhdf.error.HDF4Error:
        return []

    h, w, c = x.shape
    if patch_size == h:
        return x[np.newaxis], y[np.newaxis]

    r = list(range(0, h, patch_size))
    r[-1] = h - patch_size
    samples_x = [x[np.newaxis,i:i+patch_size, j:j+patch_size] for i in r for j in r]
    samples_y = [y[np.newaxis,i:i+patch_size, j:j+patch_size] for i in r for j in r]

    samples_x = np.concatenate(samples_x, 0)
    samples_y = np.concatenate(samples_y, 0)

    return samples_x, samples_y

def sample_generator(x, patch_size=256):
    # x = "year", "dayofyear", "hour", "minute", "v", "h", "fileahi05", "fileahi12"
    ahi05_samples, ahi12_samples = file_examples(x[6], x[7], patch_size)
    output = []
    if (len(ahi05_samples) > 0) and (len(ahi05_samples) == len(ahi12_samples)):
        finite_samples = np.all(np.isfinite(ahi05_samples), axis=(1,2,3))
        land_samples = np.nanmean(np.isfinite(ahi12_samples), axis=(1,2,3)) > 0.25
        select_samples = finite_samples * land_samples
        ahi05_samples = ahi05_samples[select_samples]
        ahi12_samples = ahi12_samples[select_samples]
        for j, sample in enumerate(ahi05_samples):
            output.append({'year': x[0], 
                           'dayofyear': x[1], 
                           'hour': x[2], 
                           'minute': x[3], 
                           'v': x[4],
                           'h': x[5],
                           'fileahi05': x[6], 
                           'fileahi12': x[7],
                           'AHI05': ahi05_samples[j].astype(np.float32), 
                           'sample_id': j, 
                           'AHI12': ahi12_samples[j].astype(np.float32), 
                          })  
    return output

def generate_maiac_dataset(ahi05_path, ahi12_path, output_url,
                     year=2018, max_files=100000, dayofyear=None,
                     subset=None):
    rowgroup_size_mb = 256

    spark = SparkSession.builder.config('spark.driver.memory', '4g').master('local[8]').getOrCreate()
    sc = spark.sparkContext

    geo = maiac_data.MAIACPairs(ahi05_path, ahi12_path)
    files = geo.paired_files(year=year, dayofyear=dayofyear)
    #idxs = np.random.randint(0, files.shape[0], min([files.shape[0], ]))
    files = files[:max_files]
    files = files.reset_index()

    print(files.columns)
    
    with materialize_dataset(spark, output_url, MAIACSchema256, rowgroup_size_mb):
        filerdd = spark.createDataFrame(files)\
             .select("year", "dayofyear", "hour", "minute", "v", "h", "fileahi05", "fileahi12")\
             .rdd.map(tuple)\
             .flatMap(sample_generator)\
             .map(lambda x: dict_to_spark_row(MAIACSchema256, x))

        spark.createDataFrame(filerdd, MAIACSchema256.as_spark_schema())\
            .coalesce(50) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)

parser = argparse.ArgumentParser()
#parser.add_argument('ahi05_path', type=str)
#parser.add_argument('ahi12_path', type=str)
#parser.add_argument('output_url', type=str)
parser.add_argument('--year', type=int, default=2018)
parser.add_argument('--max_files', type=int, default=1000)
args = parser.parse_args()


hw_data_path = '/nex/projects/goesscratch/weile/AHI05_java/output/HDF4'
maiac_data_path = '/nex/projects/goesscratch/weile/AHI_MAIAC/ver2019/data/Output'
output_url = 'file:///nobackupp10/tvandal/data/petastorm/AHI05_AHI12_256/'

generate_maiac_dataset(hw_data_path, maiac_data_path, output_url, max_files=args.max_files, year=args.year)
