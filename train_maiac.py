'''
Training script to emulate maiac using a CNN
'''
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader

from torchvision import transforms
import torch
from models import emulator

import time
import numpy as np
import random

import utils


class _transform_row:
    def __init__(self, sensor):
        mu, sd = utils.get_sensor_stats(sensor)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mu, sd),
                                             ])
    def __call__(self, row):
        AHI05 = self.transform(row['AHI05'])
        AHI12 = row['AHI12']
        AHI12 = torch.Tensor(np.transpose(AHI12, (2,0,1)))

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rotate = random.random() < 0.5
        if hflip:
            AHI05 = torch.flip(AHI05, [1,2])
            AHI12 = torch.flip(AHI12, [1,2])
        if vflip:
            AHI05 = torch.flip(AHI05, [2,1])
            AHI12 = torch.flip(AHI12, [2,1])
        if rotate:
            AHI05 = torch.rot90(AHI05, 1, [1,2])
            AHI12 = torch.rot90(AHI12, 1, [1,2]) 

        return {'AHI05': AHI05,
                'AHI12': AHI12}

def train_net(params):
    trainer = emulator.MAIACTrainer(params)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.to(device)
    trainer.load_checkpoint()

    # data transformatoins
    drop_columns = ['year', 'dayofyear', 'hour', 'minute', 'fileahi05', 'fileahi12', 'h', 'v', 'sample_id']
    transform = TransformSpec(_transform_row('AHI'), removed_fields=drop_columns)
    while 1:
        with DataLoader(make_reader(params['data_url'], num_epochs=1,
                                    transform_spec=transform), batch_size=8) as loader:
            for example in loader:
                x = example['AHI05'][:,:4].to(device)
                ahi12 = example['AHI12'].type(torch.FloatTensor)
                mask = (ahi12 != ahi12).type(torch.FloatTensor) # null values = 1

                mask = mask.to(device)
                ahi12 = ahi12.to(device)
                log = False


                if trainer.global_step % params['log_iter'] == 0:
                    log = True

                loss = trainer.step(x, ahi12, mask, log=log)
                if log:
                    print(f"Step: {trainer.global_step}\tLoss: {loss.item():4.4g}")

                if trainer.global_step % params['checkpoint_step'] == 1:
                   trainer.save_checkpoint()

                if trainer.global_step >= params['max_iter']:
                    break


if __name__ == '__main__':
    params = {'data_url': 'file:///nobackupp10/tvandal/data/petastorm/AHI05_AHI12_256/',
              'model_path': 'experiments/emulator/maiac_0.3/',
              'lr': 1e-4,
              'max_iter': 50000,
              'checkpoint_step': 1000,
              'log_iter': 100}
    train_net(params)



