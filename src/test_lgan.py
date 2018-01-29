



import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb



import tflearn
from lgan import D, G, generate_noise

import os.path as osp
import tensorflow as tf
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph






top_out_dir = '../data/'                        # Use to write Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.


model_dir = osp.join(top_out_dir, 'single_class_ae')
experiment_name = 'single_class_ae'
n_pc_points = 2048                              # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = 'emd'                             # Loss to optimize: 'emd' or 'chamfer'
class_name = "airplane"
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)    # e.g. /home/yz6/code/latent_3d_points/data/shape_net_core_uniform_samples_2048/02691156
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)


train_dir = create_dir(osp.join(top_out_dir, experiment_name))
out_dir = create_dir(osp.join(top_out_dir, "generated_planes"))
train_params = default_train_params()
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            train_dir = train_dir,
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5    

reset_tf_graph()
pdb.set_trace()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(model_dir, 500)

model_path = '../data/lgan_plane/G_network_299.pth'
g = torch.load(model_path)
batch_size=50

for i in xrange(100):
    noise=Variable(torch.cuda.FloatTensor(batch_size, 128))
    generate_noise(noise)
    fake_x = g(noise).data.cpu().numpy()
    fake_pc = ae.decode(fake_x)
    print fake_pc.shape
