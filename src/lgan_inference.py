

import torch
import torch.nn as nn
from torch.autograd import Variable
from lgan import G, D, generate_noise
import numpy as np
import sys, os
import pdb

model_path = sys.argv[1] # model path
save_file_path = sys.argv[2]  # save file at #'../data/generated_planes'

g = torch.load(model_path, map_location=lambda storage, loc: storage)
batch_size=100


noise=Variable(torch.FloatTensor(batch_size, 128))
generate_noise(noise)
fake_x = g(noise).data.numpy()


np.save(save_file_path, fake_x)
