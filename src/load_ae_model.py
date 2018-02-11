import tensorflow as tf
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder, unpickle_data
from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import apply_augmentations
from latent_3d_points.external.structural_losses.tf_nndistance import nn_distance


import sys, os
import os.path as osp
import numpy as np
import pdb





top_out_dir = '../data/'                        # Use to write Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

def load(model_path, z_rotate, num_points):
	model_dir = osp.dirname(model_path)
	model_epoch = int(osp.basename(model_path).split('-')[1])
	experiment_name = osp.basename(osp.dirname(model_path)).split('train_')[1] #'single_class_ae_plane_chamfer_z_rotate'                         # Number of points per model.
	bneck_size = 128                                # Bottleneck-AE size
	ae_loss = 'chamfer'                             # Loss to optimize: 'emd' or 'chamfer'
	class_name = "airplane"
	syn_id = snc_category_to_synth_id()[class_name]
	class_dir = osp.join(top_in_dir , syn_id)    # e.g. /home/yz6/code/latent_3d_points/data/shape_net_core_uniform_samples_2048/02691156

	train_dir = create_dir(osp.join(top_out_dir, experiment_name))
	train_params = default_train_params()
	encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(num_points, bneck_size)


	conf = Conf(n_input = [num_points, 3],
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
	            experiment_name = experiment_name,
	            allow_gpu_growth = True
	           )
	# pdb.set_trace()
	reset_tf_graph()
	ae = PointNetAutoEncoder(conf.experiment_name, conf)
	ae.restore_model(model_dir, model_epoch)
	return ae, conf

def unpickle_pc_file(data_file): 
	"""
	data_file: input pickled file of PointCloud instance
	"""
	for x in unpickle_data(data_file):
	    pc_data = x
	return pc_data

def get_feed_data(pc_data, conf):
	data, _, _ = pc_data.full_epoch_data(shuffle=False)
	feed = apply_augmentations(data, conf)
	return feed

def transform(pc_data, conf, batch_size):
	n_examples = pc_data.num_examples
	n_batch = n_examples/batch_size
	result = list()
	for _ in xrange(n_batch):
	    feed_pc, feed_model_names, _ = pc_data.next_batch(batch_size)
	    if z_rotate == 'True':
	        feed_pc = apply_augmentations(feed_pc, conf)
	    hidden = ae.transform(feed_pc)
	    result.append(hidden)

	result = np.concatenate(result, axis=0)
	return result





def reconstruct(pc_data, conf, batch_size):
	n_examples = pc_data.num_examples
	n_batch = n_examples/batch_size
	result = list()
	for _ in xrange(n_batch):
	    feed_pc, feed_model_names, _ = pc_data.next_batch(batch_size)
	    if z_rotate == 'True':
	        feed_pc = apply_augmentations(feed_pc, conf)
	    rec, _ = ae.reconstruct(feed_pc)
	    result.append(rec)

	result = np.concatenate(result, axis=0)
	return result
