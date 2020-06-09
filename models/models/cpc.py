import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

import os, sys
import numpy as np
import logging

#from .unit_networks import ContentEncoder, Decoder
from .autoencoder import Encoder, Decoder
import utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def deprocess(x):
    x = x * 0.5 + 0.5
    x[x < 0] = 0.
    x[x > 1] = 1.
    return x

def random_linear_shuffle(x):
    c = x.shape[1]
    M = torch.rand((c,c)).cuda()
    M = M.view((c,c,1,1))
    newimg = torch.nn.functional.conv2d(x, M)
    mx = torch.max(newimg)
    newimg /= mx
    return newimg


class CPC(nn.Module):
    def __init__(self, params):
        super(CPC, self).__init__()
        n_down = params['downsample']
        view_dims = params['view_dims']
        n_res = params['res_blocks']
        dim = params['encoded_dim']

        #self.encs = [ContentEncoder(n_down, n_res, input_dim, dim, 'none', 'lrelu', pad_type='zero')
        #             for input_dim in view_dims]
        self.encs = [Encoder(input_dim, dim) for input_dim in view_dims]
        self.encs = nn.ModuleList(self.encs)

    def encode(self, x, view):
        return self.encs[view](x)

    def forward(self, x, view):
        return self.encode(x, view)

class ColorizeCPC(CPC):
    def __init__(self, params):
        super(ColorizeCPC, self).__init__(params)
        n_down = params['downsample']
        view_dims = params['view_dims']
        n_res = params['res_blocks']
        dim = params['encoded_dim']

        #self.decs = [Decoder(n_down, n_res, e.output_dim, input_dim, res_norm='none', activ='lrelu', pad_type='zero')
        #            for input_dim, e in zip(view_dims, self.encs)]
        # self.decs= [Decoder(dim, input_dim) for input_dim in view_dims]
        # self.decs = nn.ModuleList(self.decs)
        self.conv_in = nn.Conv2d(dim, 128, 3, padding=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(128, 128, 1, padding=0, stride=1)

        self.conv_grey = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.conv_3 = nn.Conv2d(128+16, 3, 3, padding=1, stride=1)

    def decode(self, x_enc, grey):
        x = self.conv_in(x_enc)
        x = self.lrelu(x)
        x = self.conv_2(x)
        x = self.lrelu(x)

        x2 = self.conv_grey(grey)
        x2 = self.lrelu(x2)

        x = torch.cat((x, x2), dim=1)
        return self.conv_3(x)

    def forward(self, x, grey):
        x_enc = self.encode(x, 1)
        return self.decode(x_enc, grey)



class CPCSpatialLoss(nn.Module):
    def __init__(self, n_views=2):
        super(CPCLoss, self).__init__()
        self.loss_recon = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.n_views = n_views
        self.sigmoid = nn.Sigmoid()

    def contrast(self, z1, z2):
        zdot = z1 * z2 #(N, D, H, W)
        # sum over D x H x W
        zmean = torch.mean(zdot, (1,2,3)) #(N,)
        #probs = self.sigmoid(zmean)
        return self.sigmoid(zmean)

    def forward(self, V1, V2):
        N = V1.shape[0]
        # negative sample distances
        z_probs = torch.zeros((N,N), dtype=torch.float32).cuda()
        for i in range(N):
            z_i = V1[i:i+1] # anchor
            z_probs[i] = self.contrast(z_i, V2)

        # p(v1,v2) = diagonal
        labels = torch.eye(N).cuda()

        # cross entropy loss between positive and negative samples
        loss_bce = self.bce_loss(z_probs, labels)
        return loss_bce

class CPCLoss(nn.Module):
    def __init__(self, n_views=2):
        super(CPCLoss, self).__init__()
        self.loss_recon = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.n_views = n_views
        self.sigmoid = nn.Sigmoid()

    def contrast(self, z1, z2):
        zdot = z1 * z2 #(N, D, H, W)
        # sum over D x H x W
        zmean = torch.mean(zdot, (1,2,3)) #(N,)
        #probs = self.sigmoid(zmean)
        return self.sigmoid(zmean)

    def forward(self, V1, V2, negatives):
        N = V1.shape[0]

        # positive example
        positive_probs = self.contrast(V1, V2)

        # negative examples
        negative_probs = torch.zeros((N,len(negatives)), dtype=torch.float32).cuda()
        for i, x in enumerate(negatives):
            negative_probs[:,i] = self.contrast(V1, x)

        # concatenate probs 
        all_probs = torch.cat([positive_probs.unsqueeze(1), negative_probs],1)
        # labels 
        labels = torch.zeros_like(all_probs, dtype=torch.float32)
        labels[:,0] = 1.

        # cross entropy loss between positive and negative samples
        loss_bce = self.bce_loss(all_probs, labels)
        return loss_bce


class CPCTrainer(nn.Module):
    def __init__(self, params):
        super(CPCTrainer, self).__init__()
        self.params = params
        self.lambda_cpc = params['lambda_cpc']
        self.model = CPC(params)
        self.view_dims = params['view_dims']
        self.N_negatives = params['N_negative']
        self.N_views = len(self.view_dims)

        assert self.N_views == 2

        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        self.global_step = 0
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, params['lr_scheduler_step'],
                                                        gamma=0.1)

        # Set generator loss 
        self.cpc_loss = CPCLoss()

        # Tensorboard writer
        exp_dir = 'batch{batch_size}_down{downsample}_encoded{encoded_dim}'.format(**params)
        self.tfwriter = SummaryWriter(os.path.join(params['model_path'],
                                                   exp_dir,'cpc_tfsummary'))
        self.checkpoint_file = os.path.join(params['model_path'], exp_dir, 'checkpoint_cpc.torch')

    def save_checkpoint(self):
        checkpoint_file = self.checkpoint_file
        state = dict(global_step=self.global_step,
                     enc_state=self.model.encs.state_dict(),
                     optimizer_state=self.optimizer.state_dict(),
                     scheduler_state=self.scheduler.state_dict())
        torch.save(state, checkpoint_file)

    def load_checkpoint(self):
        checkpoint_file = self.checkpoint_file
        if not os.path.isfile(checkpoint_file):
            _logger.info("Checkpoint does not exists: %s" % checkpoint_file)
            return
        checkpoint = torch.load(checkpoint_file)
        _logger.info("checkpoint_file: {}".format(checkpoint_file))
        self.global_step = checkpoint['global_step']
        self.model.encs.load_state_dict(checkpoint['enc_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])

    def gen_update(self, v1, v2, log=False):
        '''
        inputs: v1, v2
        generate n_negative_views of v2

        positive examples: (v1, v2)
        negative examples: (v1, v2_1), (v1, v2_2), ..., (v1, v2_N)

        encode each of the views
        cross entropy on positive vs negative views
        '''
        # generate negative examples
        negatives = [random_linear_shuffle(v2) for i in range(self.N_negatives)]

        v1_enc = self.model.encode(v1, 0)
        v2_enc = self.model.encode(v2, 1)

        encoded_negs = []
        for i, x in enumerate(negatives):
            x_enc = self.model.encode(x, 1)
            encoded_negs.append(x_enc)

        ## estimate cpc with v1, v2, and negatives
        loss_cpc = self.cpc_loss(v1_enc, v2_enc, encoded_negs)

        ## Need to apply both spatial and spectral cpc losses?
        ## Total loss
        loss = self.lambda_cpc * loss_cpc

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if log:
            v = v1
            v_1_enc = self.model.encode(v1, 0)
            grid_v = torchvision.utils.make_grid(v[:4])

            v2_all = torch.stack([v2] + negatives, 0)
            grid_v2 = torchvision.utils.make_grid(v2_all[:,0])
            self.tfwriter.add_image(f'v{i}/L', deprocess(grid_v), global_step=self.global_step)
            self.tfwriter.add_image(f'v{i}/V2', deprocess(grid_v2), global_step=self.global_step)
            self.tfwriter.add_scalar(f"loss/cpc", loss_cpc, global_step=self.global_step)
            self.tfwriter.add_scalar(f'loss/loss', loss, global_step=self.global_step)
            _logger.info(f"""Step{self.global_step} -- CPC={loss_cpc.item():3.3f}\tTotal Loss={loss.item():3.3f}""")

        self.global_step += 1

class ColorizeCPCTrainer(CPCTrainer):
    def __init__(self, params):
        super(ColorizeCPCTrainer, self).__init__(params)
        self.model = ColorizeCPC(params)
        self.global_step = 0
        self.lambda_recon = params['lambda_recon']


        # finetuning optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, params['lr_scheduler_step'],
                                                        gamma=0.1)

        # Set generator loss 
        #self.recon_loss = nn.L1Loss()
        self.recon_loss = nn.MSELoss()

        # Set cpc loss
        self.cpc_loss = CPCLoss()

        # Tensorboard writer
        exp_dir = 'batch{batch_size}_down{downsample}_encoded{encoded_dim}'.format(**params)
        self.tfwriter = SummaryWriter(os.path.join(params['model_path'],
                                                   exp_dir,
                                                  'tfsummary_colorize'))
        self.checkpoint_file_cpc = os.path.join(params['model_path'], exp_dir, 'checkpoint_cpc.torch')
        self.checkpoint_file_color = os.path.join(params['model_path'], exp_dir, 'checkpoint_colorize.torch')
        torch.save(params, os.path.join(params['model_path'], exp_dir, 'colorize_params.torch'))
        self.load_checkpoint_cpc()

    def load_checkpoint_cpc(self):
        checkpoint_file = self.checkpoint_file_cpc
        if not os.path.isfile(checkpoint_file):
            _logger.info("Checkpoint does not exists: %s" % checkpoint_file)
            raise ValueError("CPC Checkpoint does not exist")
        checkpoint = torch.load(checkpoint_file)
        self.model.encs.load_state_dict(checkpoint['enc_state'])

    def save_checkpoint(self):
        state = dict(global_step=self.global_step,
                     enc_state=self.model.encs.state_dict(),
                     #dec_state=self.model.decs.state_dict(),
                     scheduler_state=self.scheduler.state_dict(),
                     optimizer_state=self.optimizer.state_dict())
        torch.save(state, self.checkpoint_file_color)

    def load_checkpoint(self):
        checkpoint_file = self.checkpoint_file_color
        if not os.path.isfile(checkpoint_file):
            _logger.info("Checkpoint does not exists: %s" % checkpoint_file)
            return
        checkpoint = torch.load(checkpoint_file)
        self.global_step = checkpoint['global_step']
        self.model.encs.load_state_dict(checkpoint['enc_state'])
        #self.model.decs.load_state_dict(checkpoint['dec_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])

    def gen_update(self, v1, v2, log=False):
         # generate negative examples
        negatives = [random_linear_shuffle(v2) for i in range(self.N_negatives)]

        v1_enc = self.model.encode(v1, 0)
        v2_enc = self.model.encode(v2, 1)

        if self.lambda_cpc == 0:
            loss_cpc = torch.tensor(0.)
        else:
            encoded_negs = []
            for i, x in enumerate(negatives):
                x_enc = self.model.encode(x, 1)
                encoded_negs.append(x_enc)

            ## estimate cpc with v1, v2, and negatives
            loss_cpc = self.cpc_loss(v1_enc, v2_enc, encoded_negs)

        rgb_recon = self.model.decode(v2_enc, v1)
        loss_recon = self.recon_loss(v2, rgb_recon)

        '''loss_recon = 0.
        views = [v1, v2]
        views_enc = [v1_enc, v2_enc]
        recon_losses = dict()
        for j, x_enc in enumerate(views_enc):
            for i in range(0, len(views)):
                x_recon = self.model.decode(x_enc, i)
                #_logger.info(f"View: {j}, Decoder: {i} - {views[j].shape} to {views[i].shape}, Recon: {x_recon.shape}")
                recon_losses[(i,j)] = self.recon_loss(views[i], x_recon)
                loss_recon += recon_losses[(i,j)]
        '''
        # total loss
        loss = self.lambda_recon * loss_recon + self.lambda_cpc * loss_cpc

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if log:
            v = v1
            v_1_enc = self.model.encode(v, 0)
            v_color = self.model.decode(v_1_enc, v1)

            grid_v_color = torchvision.utils.make_grid(v_color[:4])
            grid_v1 = torchvision.utils.make_grid(v1[:4])

            v2_all = torch.stack([v2] + negatives, 0)
            grid_v2 = torchvision.utils.make_grid(v2_all[:,0])

            self.tfwriter.add_image(f'greyscale', deprocess(grid_v1), global_step=self.global_step)
            self.tfwriter.add_image(f'positive_and_negatives', deprocess(grid_v2), global_step=self.global_step)
            self.tfwriter.add_image(f'grey_to_colorized', deprocess(grid_v_color), global_step=self.global_step)
            self.tfwriter.add_scalar(f"loss/recon", loss_recon, global_step=self.global_step)
            self.tfwriter.add_scalar(f"loss/cpc", loss_cpc, global_step=self.global_step)
            self.tfwriter.add_scalar(f'loss/loss', loss, global_step=self.global_step)
            #for key, val in recon_losses.items():
            #    self.tfwriter.add_scalar(f'loss/{key[0]}-to-{key[1]}', val, global_step=self.global_step)

            _logger.info(f"""Step: {self.global_step} -- Recon={loss_recon.item():3.3f}\tCPC={loss_cpc.item():3.3f}\tTotal Loss={loss.item():3.3f}""")

        self.global_step += 1
