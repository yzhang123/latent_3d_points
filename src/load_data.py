
import sys, os
import os.path as osp
import numpy as np
import pdb


from latent_3d_points.src.analyze import get_nn_between_codes, get_nn_distance, get_average_distance, get_nn_chamfer, get_nn_chamfer_own, get_avg_chamfer_own, get_chamfer_permut
from latent_3d_points.src.general_utils import apply_augmentations
from latent_3d_points.external.structural_losses.tf_nndistance import nn_distance
from latent_3d_points.src.load_ae_model import load as load_ae, get_feed_data, unpickle_pc_file
from latent_3d_points.src.in_out import create_dir

model_path = sys.argv[1] # model path of ae model
train_pc_file = sys.argv[2] # .pkl file of point cloud
test_pc_file = sys.argv[3] # .pkl file of point cloud
z_rotate = sys.argv[4] # 'True' or 'False'
num_points = int(sys.argv[5])
generated_hidden_file = sys.argv[6] # gerenated .npy file by gan


ae, conf = load_ae(model_path, z_rotate, num_points)

train_pc_data = unpickle_pc_file(train_pc_file)
train_feed = get_feed_data(train_pc_data, conf)
train_hidden = ae.transform(train_feed)
train_reconstr, _ = ae.reconstruct(train_feed)

test_pc_data = unpickle_pc_file(test_pc_file)
test_feed = get_feed_data(test_pc_data, conf)
test_hidden = ae.transform(test_feed)
test_reconstr, _ = ae.reconstruct(test_feed)

generated_hidden= np.load(generated_hidden_file)
generated_reconstr = ae.decode(generated_hidden)



print("nn avg distance between train and generate")
train_code2gen, train_gen2code, train_dist_list, train_mean_nn_dist = get_nn_between_codes(train_hidden, generated_hidden)

print("train, generated, gen2code")
print(train_gen2code)



print("nn avg distance between test and generate")
test_code2gen, test_gen2code, test_dist_list, test_mean_nn_dist = get_nn_between_codes(test_hidden, generated_hidden)


print("test, generated, gen2code")
print(test_gen2code)


print("train-gen fidelity")
print(train_mean_nn_dist)



print("test-gen fidelity")
print(test_mean_nn_dist)


print("train-gen coverage")
print(len(train_code2gen.keys())*1.0/ len(train_hidden))

print("test-gen coverage")
print(len(test_code2gen.keys())*1.0/ len(test_hidden))




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
# print('nn_dist_train')
# nearest_list, nn_dist_train = get_nn_distance(train_hidden)
# print("nn_dist_train: ", nn_dist_train)

# print("chamfer train nearest average using nearest list")
# avg_nearest_train = get_chamfer_permut(train_reconstr, nearest_list)
# print(avg_nearest_train)
# print('nn_dict')
# dict_code_to_gen, dict_gen_to_code, nn_list, nn_mean = get_nn(train_hidden, test_hidden)
# print("nn_mean: ", nn_mean)
# print("nn_list")
# print(nn_list)




tmp_dir = osp.join('../data', 'tmp')
create_dir(tmp_dir)

for i in range(100):
    np.savetxt(osp.join(tmp_dir, '{0}_generated.csv'.format(i)), generated_reconstr[i], delimiter=",")
    np.savetxt(osp.join(tmp_dir, '{0}_train_nearest.csv'.format(i)), train_feed[train_gen2code[i][0]], delimiter=",")
    np.savetxt(osp.join(tmp_dir, '{0}_test_nearest.csv'.format(i)), test_feed[test_gen2code[i][0]], delimiter=",")
    np.savetxt(osp.join(tmp_dir, '{0}_train_nearest_reconstr.csv'.format(i)), train_reconstr[train_gen2code[i][0]], delimiter=",")


# for i in range(100):
#     np.savetxt(osp.join(tmp_dir, '{0}_test_reconstr.csv'.format(i)), test_reconstr[i], delimiter=",")
#     np.savetxt(osp.join(tmp_dir, '{0}_test_feed.csv'.format(i)), test_feed[i], delimiter=",")
#     np.savetxt(osp.join(tmp_dir, '{0}_test_nearest.csv'.format(i)), train_feed[dict_gen_to_code[i][0]], delimiter=",")
#     np.savetxt(osp.join(tmp_dir, '{0}_test_nearest_reconstr.csv'.format(i)), train_reconstr[dict_gen_to_code[i][0]], delimiter


