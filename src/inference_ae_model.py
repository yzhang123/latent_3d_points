import numpy as np
import os.path as osp

import tflearn

import tensorflow as tf
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder
from latent_3d_points.src.tf_utils import reset_tf_graph




import sys, os
import pdb

model = sys.argv[1] # mode namel, e.g. path single_class_ae_plane_chamfer_zrotate
save_file_path = sys.argv[2]  # save hidden.npy


top_out_dir = '../data/'                        # Use to write Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.


model_dir = osp.join(top_out_dir, model)
experiment_name = model #'single_class_ae_plane_chamfer_zrotate'
n_pc_points = 600                              # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = 'chamfer'                             # Loss to optimize: 'emd' or 'chamfer'
class_name = "airplane"
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)    # e.g. /home/yz6/code/latent_3d_points/data/shape_net_core_uniform_samples_2048/02691156
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)


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
            z_rotate = train_params['z_rotate'],
            train_dir = train_dir,
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5              # How often to evaluate/print out loss on held_out data (if any).

# pdb.set_trace()
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)


n_examples = all_pc_data.num_examples
batch_size = conf.batch_size
n_batches = int(n_examples / batch_size)
latent_list=list()
for _ in xrange(n_batches):
    feed_pc, feed_model_names, _ = all_pc_data.next_batch(batch_size)
    num_points_to_pick = n_pc_points
    perm = np.random.permutation(num_points_to_pick)
    feed_pc = np.take(feed_pc, perm[0:num_points_to_pick], axis=1)
    latent_codes = ae.transform(feed_pc)
    latent_list.append(latent_codes)
    
latent = np.concatenate(latent_list, axis=0)
ae.restore_model(model_dir, 250)

np.save(save_file_path, latent)

