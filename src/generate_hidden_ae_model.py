

import sys, os
import os.path as osp
import numpy as np
import pdb

from latent_3d_points.src.load_ae_model import load as load_ae, unpickle_pc_file, get_feed_data

model_path = sys.argv[1] # mode namel, e.g. path single_class_ae_plane_chamfer_zrotate
pc_file = sys.argv[2]
z_rotate = sys.argv[3]
num_points = int(sys.argv[4])


ae, conf = load_ae(model_path, z_rotate, num_points)
pc_data = unpickle_pc_file(pc_file)
feed = get_feed_data(pc_data, conf)

hidden = ae.transform(feed)
out_dir = os.path.dirname(pc_file)
# np.save(osp.join(out_dir, 'hidden.npy'), hidden)

