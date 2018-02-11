

import sys, os
import os.path as osp
import numpy as np
import pdb

from latent_3d_points.src.load_ae_model import load as load_ae, unpickle_pc_file, get_feed_data
from latent_3d_points.src.general_utils import apply_augmentations

model_path = sys.argv[1] # mode namel, e.g. path single_class_ae_plane_chamfer_zrotate
pc_file = sys.argv[2]
z_rotate = sys.argv[3]
num_points = int(sys.argv[4])


ae, conf = load_ae(model_path, z_rotate, num_points)
pc_data = unpickle_pc_file(pc_file)

batchsize = 100
n_examples = pc_data.num_examples
n_batch = n_examples/batchsize


out_dir = os.path.dirname(pc_file)
result = list()

for _ in xrange(n_batch):
    feed_pc, feed_model_names, _ = pc_data.next_batch(batchsize)
    if z_rotate == 'True':
        feed_pc = apply_augmentations(feed_pc, conf)
    hidden = ae.transform(feed_pc)
    result.append(hidden)

result = np.concatenate(result, axis=0)
np.save(osp.join(out_dir, 'hidden.npy'), result)





# feed = get_feed_data(pc_data, conf)

# hidden = ae.transform(feed)
# out_dir = os.path.dirname(pc_file)

