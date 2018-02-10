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
train_pc_file = sys.argv[4]
test_pc_file = sys.argv[5]
z_rotate = sys.argv[6] # 'True' or 'False'

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
# pdb.set_trace()
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(model_dir, model_epoch)

for x in unpickle_data(train_pc_file):
    train_pc_data = x
for x in unpickle_data(test_pc_file):
    test_pc_data = x

train_data, _, _ = train_pc_data.full_epoch_data(shuffle=False)
train_feed = apply_augmentations(train_data, conf)
train_hidden = ae.transform(train_feed)
train_reconstr, _ = ae.reconstruct(train_feed)

test_data, _, _ = test_pc_data.full_epoch_data(shuffle=False)
test_feed = apply_augmentations(test_data, conf)
test_hidden = ae.transform(test_feed)
test_reconstr, _ = ae.reconstruct(test_feed)


# print('chamfer distance between test and train reconstruct')
# fake_to_true, true_to_fake, dist_list = get_nn_chamfer(train_reconstr, test_reconstr)
# print("Output", np.mean(dist_list))
# print('chamfer nn distance between train reconstructions')
# fake_to_true, true_to_fake, dist_list = get_nn_chamfer_own(train_reconstr)
# print("Output", np.mean(dist_list))
# print('chamfer avg distance between train reconstructions')
# fake_to_true, true_to_fake, dist_list = get_avg_chamfer_own(train_reconstr)
# print("Output", np.mean(dist_list))
    
# print('avg_dist_train')
# avg_dist_train = get_average_distance(train_hidden)
# print("avg_dist_train: ", avg_dist_train)
print('nn_dist_train')
nearest_list, nn_dist_train = get_nn_distance(train_hidden)
print("nn_dist_train: ", nn_dist_train)

print("chamfer train nearest average using nearest list")
avg_nearest_train = get_chamfer_permut(train_reconstr, nearest_list)
print(avg_nearest_train)
# print('nn_dict')
# dict_code_to_gen, dict_gen_to_code, nn_list, nn_mean = get_nn(train_hidden, test_hidden)
# print("nn_mean: ", nn_mean)
# print("nn_list")
# print(nn_list)




# tmp_dir = osp.join(top_out_dir, 'tmp_rotate')
# create_dir(tmp_dir)

# for i in range(100):
#     np.savetxt(osp.join(tmp_dir, '{0}_test_reconstr.csv'.format(i)), test_reconstr[i], delimiter=",")
#     np.savetxt(osp.join(tmp_dir, '{0}_test_feed.csv'.format(i)), test_feed[i], delimiter=",")
#     np.savetxt(osp.join(tmp_dir, '{0}_test_nearest.csv'.format(i)), train_feed[dict_gen_to_code[i][0]], delimiter=",")
#     np.savetxt(osp.join(tmp_dir, '{0}_test_nearest_reconstr.csv'.format(i)), train_reconstr[dict_gen_to_code[i][0]], delimiter=",")


