# Spectral Synthesis for Satellite-to-Satellite Translation

[ADD Link to TGRS paper]

## Model

VAE-GAN Architecture forr unsupervised image-to-image translation with shared spectral reconstruction loss. Model is trained on GOES-16/17 and Himawari-8 L1B data processed by GeoNEX. 

![Network Architecture](images/image-to-image-sensors.png)

![Alt Text](images/synthetic_animation.gif)


## Dependencies

Python==3.7 <br>
Pytorch==1.5 <br>
Petastorm==0.9

Note: Functionality using PyTorch with MPI requires installation from source.

```
conda create --name geonex_torch1.5 python=3.7 pytorch=1.5 xarray numpy scipy pandas torchvision tensorboard opencv pyyaml jupyterlab matplotlib seaborn
conda install -c conda-forge pyhdf
pip install petastorm
```

## Steps to Reproduce Experiments

### Data

Find GOES-16/17 and Himawari-8 L1G products on the [GeoNEX dataportal](https://data.nas.nasa.gov/geonex/data.php)

### Build Training Datasets

Data is parsed from GeoNEXL1G as sub-images and stored in a petastorm database using spark. We set max_files to 100 for testing only. 

```
cd data
python write_data_to_petastorm.py /nex/datapool/geonex/public/GOES16/GEONEX-L1G/ WRITE_DIRECTORY G16 --year 2018 --max_files 100
python write_data_to_petastorm.py /nex/datapool/geonex/public/GOES17/GEONEX-L1G/ WRITE_DIRECTORY G17 --year 2018 --max_files 100
```

### Run Training script

Train model with a given configuration file with data and model parameter.  See `configs/Base-G16G17.yaml` and `configs/Base-G16G17H8.yaml` as examples.

```
python train_net.py --config_file configs/Base-G16G17.yaml
```

Training can be visualized using tensorboard
```
tensorboard --logdir EXPERIMENT_DIRECTORY
```

### Perform Inference

Work in progress <br>
Current inference examples can be found in notebooks/


### Known Challenges

This model estimated the lower bound of log-likelihood effectively causing reduced spatial resolution. The latent space is only appoximately cycle consistent. Recent developed in invertible methods (eg. AlignFlow) solves this problem deterministically with maximum likelihood.

### Acknowledgements 

This work was funded by the NASA Ames Research Center and NASA Earth eXchange (NEX). We acknowledge the network codes inherented from https://github.com/mingyuliutw/UNIT.
