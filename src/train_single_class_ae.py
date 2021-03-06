import os.path as osp
import sys

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder, pickle_data, save_csv

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import apply_augmentations

top_out_dir = '../data/'                        # Use to write Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = sys.argv[1]  #' '
n_pc_points = int(sys.argv[2]) #600  # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = sys.argv[3] #'chamfer'                             # Loss to optimize: 'emd' or 'chamfer'
class_name = sys.argv[4] #'airplane'
z_rotate = sys.argv[5] # 'True' or 'False'
fixed_points = sys.argv[6] #'True' or 'False'

syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)

# point cloud instance
train_pc, val_pc, test_pc = load_all_point_clouds_under_folder(class_dir, n_threads=2, file_ending='.ply', verbose=True, fixed_points=fixed_points == 'True', num_points=n_pc_points)
train_dir = create_dir(osp.join(top_out_dir, 'train_'+experiment_name))
val_dir = create_dir(osp.join(top_out_dir, 'val_'+experiment_name))
test_dir = create_dir(osp.join(top_out_dir, 'test_'+experiment_name))

pickle_data(osp.join(train_dir, 'train_pc.pkl'), train_pc)
pickle_data(osp.join(val_dir, 'val_pc.pkl'), val_pc)
pickle_data(osp.join(test_dir, 'test_pc.pkl'), test_pc)


# dictionary
train_params = default_train_params()
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)


conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = 1000, #train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            test_dir = test_dir,
            val_dir = val_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = z_rotate == 'True', #train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args,
            experiment_name = experiment_name,
            val_step = 5,
            test_step = 200
           )
            # How often to evaluate/print out loss on held_out data (if any). # epochs
conf.save(osp.join(train_dir, 'configuration'))

reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

buf_size = 1 # flush each line
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(train_pc, conf, log_file=fout, val_data=val_pc, test_data=test_pc)
fout.close()

print('On train hidden transform')
train_hidden, _, _ = train_pc.full_epoch_data()
train_hidden = apply_augmentations(train_hidden, conf)
train_hidden = ae.transform(train_hidden)
np.save(osp.join(train_dir, 'hidden.npy'), train_hidden)

print('On val hidden transform')
val_hidden, _, _ = val_pc.full_epoch_data()
val_hidden = apply_augmentations(val_hidden, conf)
val_hidden = ae.transform(val_hidden)
np.save(osp.join(val_dir, 'hidden.npy'), val_hidden)


print('On test hidden transform')
test_hidden, _, _ = test_pc.full_epoch_data()
test_hidden = apply_augmentations(test_hidden, conf)
test_hidden = ae.transform(test_hidden)
np.save(osp.join(test_dir, 'hidden.npy'), test_hidden)



print('On train data reconstruction')
reconstructions, data_loss, feed_data, label_ids, original_data = ae.evaluate(train_pc, conf)
save_csv(osp.join(c.train_dir, 'reconstr_epoch_%s' % conf.training_epochs), reconstructions, label_ids, max_to_save=100)
save_csv(osp.join(c.train_dir, 'feeddata_epoch_%s' % conf.training_epochs), feed_data, label_ids, max_to_save=100)
