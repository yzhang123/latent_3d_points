

import torch
import torch.nn as nn
from torch.autograd import Variable
from lgan import G, D, generate_noise
import numpy as np
import sys, os
import pdb
import os.path as osp

model_path = sys.argv[1] # model path
save_file_path = osp.join(osp.dirname(model_path), 'hidden_generated.npy')
g = torch.load(model_path, map_location=lambda storage, loc: storage)
num_objects = 100000
batch_size=100



num_iters = num_objects/batch_size

fake_list = list()
noise=Variable(torch.FloatTensor(batch_size, 128))

for _ in range(num_iters):
	generate_noise(noise)
	fake_z = g(noise).data.numpy()
	fake_list.append(fake_z)
fake_z = np.concatenate(fake_list, axis=0)
np.save(save_file_path, fake_z)






# gan_model_path = '/home/yz6/code/latent_3d_points/data/lgan_single_class_ae_plane_chamfer_nonrotate_600p/G_network_990.pth' # model path
# ae_model_path = '/home/yz6/code/latent_3d_points/data/train_single_class_ae_plane_chamfer_nonrotate_600p/models.ckpt-990' # model path


# ae = load_ae(ae_model_path, zrotate='False', num_points=600)

# g = torch.load(gan_model_path, map_location=lambda storage, loc: storage)
# batch_size=100
# noise=Variable(torch.FloatTensor(batch_size, 128))
# generate_noise(noise)
# fake_z = g(noise).data.numpy()

# fake_x = ae.decode(fake_z)