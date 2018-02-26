



import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb


from lgan import D, G, generate_noise

import os.path as osp

import sys


model_path = '/home/yz6/code/latent_3d_points/data/lgan_simplegan_ae_chair_chamfer/G_network_100.pth'

save_dir = osp.dirname(model_path)
g = torch.load(model_path)
batch_size=50

fake_z = list()
for i in xrange(1000):
    noise=Variable(torch.cuda.FloatTensor(batch_size, 128))
    generate_noise(noise)
    fake_x = g(noise).data.cpu().numpy()
    fake_z.append(fake_x)

fake_z = np.concatenate(fake_z, axis=0)
np.save(osp.join(save_dir, 'hidden.npy'), fake_z)
        
        

