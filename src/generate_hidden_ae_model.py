import tensorflow as tf
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder, unpickle_data
from latent_3d_points.src.tf_utils import reset_tf_graph
from analyze import get_nn, get_nn_distance, get_average_distance, get_nn_chamfer, get_nn_chamfer_own, get_avg_chamfer_own, get_chamfer_permut

from latent_3d_points.src.general_utils import apply_augmentations
from latent_3d_points.external.structural_losses.tf_nndistance import nn_distance


import sys, os
import os.path as osp
import numpy as np
import pdb

model = sys.argv[1] # mode namel, e.g. path single_class_ae_plane_chamfer_zrotate
model_epoch = int(sys.argv[2])
n_pc_points = int(sys.argv[3])
pc_file = sys.argv[4]
z_rotate = sys.argv[5] # 'True' or 'False'

top_out_dir = '../data/'                        # Use to write Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.


model_dir = osp.join(top_out_dir, model)
experiment_name = model.split('train_')[1] #'single_class_ae_plane_chamfer_zrotate'                         # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = 'chamfer'                             # Loss to optimize: 'emd' or 'chamfer'
class_name = "airplane"
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)    # e.g. /home/yz6/code/latent_3d_points/data/shape_net_core_uniform_samples_2048/02691156

train_dir = create_dir(osp.join(top_out_dir, experiment_name))
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
            z_rotate = z_rotate == 'True',
            train_dir = train_dir,
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args,
            experiment_name = experiment_name
           )


reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(model_dir, model_epoch)

for x in unpickle_data(pc_file):
    pc_data = x

data, _, _ = pc_data.full_epoch_data(shuffle=False)
feed = apply_augmentations(data, conf)
hidden = ae.transform(feed)

out_dir = os.path.dirname(pc_file)
np.save(osp.join(out_dir, 'hidden.npy'), hidden)

