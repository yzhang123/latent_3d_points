import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder_split

from latent_3d_points.src.tf_utils import reset_tf_graph


top_out_dir = '../data/'                        # Use to write Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'single_class_ae_chair_chamfer2'
n_pc_points = 2048                              # Number of points per model.
bneck_size = 128                                # Bottleneck-AE size
ae_loss = 'chamfer'                             # Loss to optimize: 'emd' or 'chamfer'
class_name = 'chair'

syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)
train_pc_data, test_pc_data, val_pc_data = load_all_point_clouds_under_folder_split(class_dir, n_threads=2, file_ending='.ply', verbose=True)


train_dir = create_dir(osp.join(top_out_dir, experiment_name))
train_params = default_train_params()
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = 2000, #train_params['training_epochs'],
            batch_size = 50,
            denoising = train_params['denoising'],
            learning_rate = 0.0005,
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = False, #train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5              # How often to evaluate/print out loss on held_out data (if any).

conf.save(osp.join(train_dir, 'configuration'))

reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)


buf_size = 1 # flush each line
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
fout.write('training on chairs, epochs=2000, bs=50, lr=0.0005, chamfer_loss, using 85/5/10 split\n')
train_stats = ae.train(train_pc_data, conf, log_file=fout, held_out_data=val_pc_data)
fout.close()