import torch
from torch import nn
from torch.autograd import Variable
from .unit_networks import ContentEncoder, Decoder, ResBlocks

class EncoderSmall(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_ch, 128, 3, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 1, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, out_ch, 3, padding=0, stride=1),
        )

    def forward(self, x):
        return self.model(x)

class DecoderSmall(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 1, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, out_ch, 3, padding=0, stride=1),
        )

    def forward(self, x):
        return self.model(x)

class SplitAutoEncoder(nn.Module):
    def __init__(self, params):
        super(SplitAutoEncoder, self).__init__()
        enc_dim = params['dim']
        in_ch_list = params['input_dims']
        self.encoders = nn.ModuleList([EncoderSmall(in_ch, 64) for in_ch in in_ch_list])
        self.shared = nn.Sequential(
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(64, enc_dim, 1, padding=0, stride=1),
                        nn.LeakyReLU(0.2))
        self.decoders = nn.ModuleList([DecoderSmall(enc_dim, in_ch) for in_ch in in_ch_list])

    def _encode(self, x, i):
        enc = self.encoders[i](x)
        enc = self.shared(enc)
        noise = Variable(torch.randn(enc.size()).cuda(enc.data.get_device()))
        return enc, noise

    def encode(self, x_list):
        encs = [self.encoders[i](x) for i, x in enumerate(x_list)]
        shared = [self.shared(enc) for enc in encs]
        noise = [Variable(torch.randn(x.size()).cuda(x.data.get_device())) for x in shared]
        return shared, noise

    def decode(self, z_list):
        decs = [self.decoders[i](x) for i, x in enumerate(z_list)]
        return decs

    def forward(self, x_list):
        z_list, _ = self.encode(x_list)
        return self.decode(z_list)

class SplitGenVAE(nn.Module):
    def __init__(self, params):
        super(SplitGenVAE, self).__init__()
        self.params = params
        enc_dim = params['gen']['dim']
        n_downsample = params['gen']['n_downsample']
        n_res = params['gen']['n_res']
        activ = params['gen']['activ']
        pad_type = params['gen']['pad_type']
        
        encoders = dict()
        decoders = dict()
        for name, item in params['data'].items():
            encoders[name] = ContentEncoder(n_downsample, n_res, item['dim'], enc_dim, 'none', activ, pad_type=pad_type)
            decoders[name] = Decoder(n_downsample, n_res, enc_dim, item['dim'], res_norm='none', activ=activ, pad_type=pad_type)
            
        self.names = encoders.keys()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.shared = ResBlocks(1, enc_dim, norm='none', activation='relu')

    def encode(self, x, name):
        enc = self.encoders[name](x)
        enc = self.shared(enc)
        noise = Variable(torch.randn(enc.size()).cuda(enc.data.get_device()))
        return enc, noise

    def decode(self, z, name):
        return self.decoders[name](z)
    
    def forward(self, x, name):
        z, _ = self.encode(x, name)
        return self.decode(z, name)

    
    def add_domain(self, name, dim):
        enc_dim = self.params['gen']['dim']
        n_downsample = self.params['gen']['n_downsample']
        n_res = self.params['gen']['n_res']
        activ = self.params['gen']['activ']
        pad_type = self.params['gen']['pad_type']
        
        encoder = ContentEncoder(n_downsample, n_res, dim, enc_dim, 'none', activ, pad_type=pad_type)
        decoder = Decoder(n_downsample, n_res, enc_dim, dim, res_norm='none', activ=activ, pad_type=pad_type)
        self.encoders.update({name: encoder})
        self.decoders.update({name: decoder})