
import sys, os
import os.path as osp
import numpy as np
import pdb


from latent_3d_points.src.general_utils import apply_augmentations
from latent_3d_points.src.load_ae_model import load as load_ae, unpickle_pc_file, get_feed_data

model_path = sys.argv[1] # mode namel, e.g. path single_class_ae_plane_chamfer_zrotate
pc_file = sys.argv[2]
z_rotate = sys.argv[3] # 'True' or 'False'
num_points = int(sys.argv[4])

ae, conf = load_ae(model_path, z_rotate, num_points)
pc_data = unpickle_pc_file(pc_file)
feed = get_feed_data(pc_data, conf)
hidden = ae.transform(feed)
reconstr, _ = ae.reconstruct(feed)


file_idx = 0
for _ in xrange(100):
    feed_pc, feed_model_names, _ = pc_data.next_batch(1)

    if z_rotate == 'True':
        feed_pc = apply_augmentations(feed_pc, conf)
    fake_pc, _ = ae.reconstruct(feed_pc)
    for real, fake in zip(feed_pc, fake_pc):
        np.savetxt(os.path.join(save_dir, "{0}_{1}_{2}".format(class_name, file_idx, "real")), real, delimiter=",")
        np.savetxt(os.path.join(save_dir, "{0}_{1}_{2}".format(class_name, file_idx, "fake")), fake, delimiter=",")
        file_idx += 1



