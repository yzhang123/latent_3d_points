
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

model = sys.argv[1] # mode name, e.g. single_class_ae_plane_chamfer_zrotate
hidden_code_file = sys.argv[2]
save_file_path = sys.argv[3]  # save file at #'../data/generated_planes'















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
conf.held_out_step = 5              # How often to evaluate/print out loss on held_out data (if any).

# pdb.set_trace()
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(model_dir, 700)

save_dir  = save_file_path #os.path.dirname(save_file_path)
if not os.path.exists(save_dir):
      os.makedirs(save_dir)

# # INTERPOLATE 

# #==========================================================
# # c1 = np.array([0.4210078 , 2.1664896 , 0.59920925, 0.28140515, 0.21958666,
# #        0.8362925 , 0.29658416, 1.2453814 , 1.7225889 , 0.71559423,
# #        0.5358459 , 0.89340943, 1.1798401 , 0.91103554, 0.20582218,
# #        0.8133202 , 0.5531029 , 0.54211384, 0.5179516 , 0.5028096 ,
# #        0.6457422 , 0.2412121 , 0.66515416, 0.6880503 , 0.59401023,
# #        0.71048635, 0.21723714, 0.8368634 , 0.10245515, 2.1405115 ,
# #        0.44162056, 0.2712193 , 2.6740043 , 0.3286272 , 0.9804301 ,
# #        0.35390428, 0.33689177, 0.26536405, 0.15394446, 0.24351865,
# #        0.8386102 , 1.04444   , 0.90032095, 0.9196596 , 2.10678   ,
# #        0.50021803, 1.3643095 , 1.01788   , 2.0179813 , 2.0501409 ,
# #        0.15902859, 0.97216284, 1.3035741 , 0.7713703 , 0.18865706,
# #        0.5560428 , 0.45678377, 0.32549924, 0.3557852 , 0.6218606 ,
# #        0.48770827, 0.44530404, 0.45654273, 0.8217225 , 1.9192965 ,
# #        0.30682975, 0.5040995 , 0.00999734, 0.12089913, 0.37829012,
# #        0.29116827, 0.737285  , 0.12179986, 0.9012906 , 0.49803227,
# #        0.4143892 , 1.7191682 , 0.22141074, 0.24530727, 0.52295965,
# #        0.20214671, 0.76002526, 1.3748028 , 0.59064496, 0.16778061,
# #        0.8156178 , 0.8520418 , 0.37554428, 1.5734951 , 0.6439508 ,
# #        0.8495066 , 0.21972746, 0.19128057, 1.4797909 , 0.9233383 ,
# #        0.3426739 , 0.16495745, 0.24916859, 0.7572104 , 0.7029389 ,
# #        0.17473042, 0.4783784 , 0.3140668 , 0.4692129 , 1.1581335 ,
# #        1.0066911 , 0.86137694, 2.5781147 , 0.31698245, 1.2424059 ,
# #        0.31985617, 0.30367994, 0.09062716, 0.5552815 , 1.584631  ,
# #        1.2427332 , 0.36141837, 0.21310008, 0.5305862 , 0.24895054,
# #        1.2359527 , 3.743509  , 0.5506564 , 0.44449812, 0.28206518,
# #        0.2982634 , 0.27564687, 0.45045498], dtype=np.float32)

# # c2 = np.array([0.3677808 , 0.37806   , 0.3651443 , 0.32089996, 0.774289  ,
# #        0.5157719 , 0.7753552 , 0.5473328 , 0.29564464, 0.5205766 ,
# #        0.15674046, 0.23024875, 0.45354447, 0.4300284 , 1.2945182 ,
# #        0.29353622, 0.26282945, 0.38322923, 0.42497274, 1.0315123 ,
# #        0.5263029 , 0.15908895, 0.6404295 , 0.34377673, 0.29857376,
# #        0.44822878, 1.6599703 , 0.6275846 , 0.51611084, 0.6233314 ,
# #        0.8793265 , 0.30876726, 0.15862645, 0.44707286, 1.3924216 ,
# #        1.2407098 , 0.4031448 , 0.2922232 , 0.74430066, 0.6769072 ,
# #        1.479191  , 0.3239333 , 1.3793646 , 1.3431895 , 0.39266837,
# #        0.93610126, 0.166745  , 0.46881142, 0.19177707, 0.5809226 ,
# #        1.373226  , 0.70915794, 2.1308324 , 1.0454025 , 2.1382618 ,
# #        1.1151115 , 1.0393177 , 0.3331157 , 0.254475  , 0.19752786,
# #        1.9489373 , 0.9589918 , 0.41583684, 0.24078262, 0.77466255,
# #        1.0779166 , 0.43465328, 0.24648215, 0.39847112, 1.1756501 ,
# #        0.83172506, 1.2409018 , 0.24571782, 0.7340951 , 0.27245325,
# #        0.6339935 , 0.5924904 , 0.15714268, 0.73309916, 0.5487076 ,
# #        0.15641955, 1.0765435 , 0.60071796, 0.35180736, 0.1744023 ,
# #        0.2178194 , 1.9140261 , 0.51999855, 0.19354904, 1.0818737 ,
# #        1.2572291 , 0.17874569, 0.46190518, 0.14212479, 1.1691449 ,
# #        0.43790662, 1.0531625 , 0.3820198 , 0.35309574, 0.26690364,
# #        0.12140626, 0.47725394, 0.769816  , 0.54084486, 0.6574315 ,
# #        0.4617548 , 0.38026154, 0.25951937, 0.545347  , 0.40907764,
# #        0.8811328 , 0.27886802, 0.10813654, 0.6021475 , 0.7054772 ,
# #        1.0566816 , 0.37540013, 0.2925203 , 0.23523986, 0.3713524 ,
# #        0.39155352, 0.554605  , 0.54650015, 0.54756767, 0.48258168,
# #        0.5612983 , 0.41828984, 0.11869045], dtype=np.float32) 

# # add = np.linspace(0, 1, 20).reshape(20, 1) #std at 0.''5

# # c1 = np.tile(c1, (20, 1)).reshape(20, -1)
# # c2 = np.tile(c2, (20, 1)).reshape(20, -1)

# # fake_x = add * c1 + (1-add) * c2

# #===============================================================





# #=============================== noise around one code=====================
# # fake_x = np.array([0.4956836 , 0.5355693 , 0.31649613, 0.16300055, 0.31800008,
# #        1.463437  , 0.2783882 , 0.64873517, 0.4974432 , 0.7169898 ,
# #        0.23554799, 0.45615435, 0.17911656, 0.4668146 , 0.15865655,
# #        0.2798081 , 0.25834778, 0.45858973, 0.40883407, 0.75517166,
# #        0.38540173, 0.09347998, 0.87695277, 0.23641519, 0.17335117,
# #        0.6091405 , 0.72566634, 0.6885015 , 0.10629082, 1.3674893 ,
# #        0.27957216, 0.2915073 , 0.3891444 , 0.32554412, 0.99169517,
# #        0.28307962, 0.23148072, 0.2698138 , 0.2273243 , 0.42533055,
# #        1.1743889 , 0.29815722, 1.2474437 , 0.7563962 , 1.41046   ,
# #        0.478049  , 0.9131854 , 0.811034  , 0.37680966, 1.2022926 ,
# #        0.31063968, 0.7881748 , 1.6014473 , 0.7423005 , 0.586867  ,
# #        0.21506244, 0.7571888 , 0.37976336, 0.28491226, 0.24426737,
# #        0.8947919 , 0.36369956, 0.36489147, 0.30646962, 1.1323605 ,
# #        0.50330865, 0.31694052, 0.08123327, 0.12279874, 0.71283716,
# #        0.35070753, 1.0826464 , 0.3773211 , 0.46725702, 0.21079826,
# #        0.34406424, 0.75763243, 0.10452541, 0.17283306, 0.4220981 ,
# #        0.17524318, 0.8796874 , 0.57207733, 0.23210014, 0.21132872,
# #        0.36693746, 1.322203  , 0.5942419 , 0.5063575 , 0.62335896,
# #        0.87713057, 0.15714793, 0.30336094, 0.45049205, 1.170471  ,
# #        0.2921643 , 0.27460763, 0.27984574, 0.4078843 , 0.485285  ,
# #        0.08815815, 0.5481875 , 0.28046566, 0.3830639 , 0.47185433,
# #        0.9839646 , 0.44628385, 0.23689553, 0.36120698, 0.3000404 ,
# #        0.25644258, 0.28463274, 0.09675838, 0.40783307, 1.2595184 ,
# #        0.80819744, 0.3746128 , 0.26964495, 0.9967914 , 0.20583394,
# #        0.31572515, 1.2708615 , 0.51022136, 0.4766037 , 0.453278  ,
# #        0.35709512, 0.4018623 , 0.18228726], dtype=np.float32)

# # fake_x = np.tile(fake_x, (20, 1)).reshape(20, -1)
# # add = np.linspace(0, 0.20, 20).reshape(20, 1) #std at 0.''5

# # fake_x = add + fake_x
# # fake_x += add


# #==============================================================================



fake_x=np.load(hidden_code_file)
fake_pc = ae.decode(fake_x[:100])


for i, x in enumerate(fake_pc):
    # path = os.path.join(save_dir, '{0}.csv'.format(i))
    # np.savetxt(os.path.join(save_dir, "0_{0}".format(add[i])), x, delimiter=",")
    np.savetxt(os.path.join(save_dir, "0_{0}".format(i)), x, delimiter=",")