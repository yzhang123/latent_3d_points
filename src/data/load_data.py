import numpy as np 
import os.path as osp
import re
import os
import pymesh
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool

import pdb



def load_obj(file_name, num_samples=2048*8):
    # print(file_name)
    # mesh = pymesh.load_mesh(file_name)
    # points = mesh.vertices
    # return points
    vertices = list()
    faces = list()
    with open(file_name, 'r') as fp:
        reader = csv.reader(fp, delimiter = ' ')
        for line in reader:
            if line[0] == 'v':
                # pdb.set_trace()
                vertices.append(np.array([float(line[1]), float(line[2]), float(line[3])]))
            elif line[0] == 'f':
                faces.append(np.array([int(line[1]), int(line[2]), int(line[3])]) - 1)
            else:
                # print(line)
                raise Exception('unexpected file format')
    
    faces_areas = map(lambda f: np.linalg.norm(np.cross(vertices[f[1]] - vertices[f[0]], vertices[f[2]] - vertices[f[0]])), faces)
    faces_areas = faces_areas / np.sum(faces_areas)
    faces_areas = np.cumsum(faces_areas)

    rand = np.random.rand(num_samples)
    indices = np.searchsorted(faces_areas, rand)
    sampled_points = map(lambda i: (lambda rand: 
            (1 - np.sqrt(rand[0]))*vertices[faces[i][0]] +
            np.sqrt(rand[0])*(1 - rand[1])*vertices[faces[i][1]] +
            np.sqrt(rand[0])*rand[1]*vertices[faces[i][2]]
        )(np.random.rand(2)),indices)
    return norm_pc_std_norm_dist(np.array(sampled_points))


# this is it#
def pc_loader(file_name):
    f_id, f_ext = osp.splitext(osp.basename(file_name))
    vertices = load_obj(file_name)
    return vertices, f_id, 'car'

def load_labels(annotation_file):

    with open(annotation_file, 'r') as fp:
        reader = csv.reader(fp)
        header = reader.next()
        colums = defaultdict(list)
        rows = defaultdict(list)
        for line in reader:
            rows[line[0]] = line[1:]
            for h, v in zip(header, line):
                columns[h].append(v)
    return rows, columns

def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name

def norm_pc_std_norm_dist(point_cloud):
    point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    return point_cloud / np.std(point_cloud)

def load_all_point_clouds_under_folder(top_dir, n_threads=20, file_ending='.ply', verbose=False, fixed_points=True, num_points=2048):
    """
    return train, val and test instances of PointCloudDataSet
    """
    file_names          = [f for f in files_in_subdirs(top_dir, file_ending)]
    num_examples        = len(file_names)
    np.random.shuffle(file_names)
    num_train_examples  = int(num_examples * 0.8)
    num_val_examples    = int(num_examples * 0.1)
    num_test_examples   = num_examples - num_train_examples - num_val_examples

    train_files = file_names[ :num_train_examples]
    val_files   = file_names[num_train_examples : num_train_examples+num_val_examples]
    test_files  = file_names[-num_test_examples: ]
    train_pclouds, train_model_ids, train_syn_ids   = load_point_clouds_from_filenames(train_files, n_threads, loader=pc_loader, verbose=verbose)
    val_pclouds, val_model_ids, val_syn_ids         = load_point_clouds_from_filenames(val_files, n_threads, loader=pc_loader, verbose=verbose)
    test_pclouds, test_model_ids, test_syn_ids      = load_point_clouds_from_filenames(test_files, n_threads, loader=pc_loader, verbose=verbose)
    # pdb.set_trace()
    return  VariableSizePointCloudDataSet(train_pclouds, labels=train_syn_ids + '_' + train_model_ids, init_shuffle=False, fixed_points=fixed_points, num_points=num_points), \
            VariableSizePointCloudDataSet(val_pclouds, labels=val_syn_ids + '_' + val_model_ids, init_shuffle=False, fixed_points=fixed_points, num_points=num_points), \
            VariableSizePointCloudDataSet(test_pclouds, labels=test_syn_ids + '_' + test_model_ids, init_shuffle=False, fixed_points=fixed_points, num_points=num_points)

def load_point_clouds_from_filenames(file_names, n_threads, loader, verbose=False):
    pc = loader(file_names[0])[0]   # numpy array num_points x 3
    pclouds = [None] * len(file_names)
    # pdb.set_trace()
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    # pool = Pool(n_threads)

    # for i, data in enumerate(pool.imap(loader, file_names)):
    for i, data in enumerate(map(loader, file_names)):
        pclouds[i], model_names[i], class_ids[i] = data #class id, id of e.g. plane

    # pool.close()
    # pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids

class VariableSizePointCloudDataSet(object):
    # def __init__(self, point_clouds, init_shuffle, num_points, fixed_points):

    def __init__(self, point_clouds, noise=None, labels=None, copy=True, init_shuffle=True, num_points=2048, fixed_points=True):
        '''Construct a DataSet.
        Args:
            init_shuffle, shuffle data before first epoch has been reached.
        Output:
            original_pclouds, labels, (None or Feed) # TODO Rename
        '''

        self.num_examples = len(point_clouds) # number of shapes
        
        if labels is not None:
            assert len(point_clouds) == labels.shape[0], ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels

        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        if noise is not None:
            if copy:
                self.noisy_point_clouds = [pc.copy() for pc in noise]
            else:
                self.noisy_point_clouds = noise
        else:
            self.noisy_point_clouds = None

        if copy:
            self.point_clouds = [pc.copy() for pc in point_clouds]
        else:
            self.point_clouds = point_clouds

        # extend point clouds to at least size num_points
        self.point_clouds = [
            np.repeat(
                pc,
                (num_points + pc.shape[0] - 1) / pc.shape[0],
                axis=0) 
            for pc in self.point_clouds]

        self.num_points = num_points # number of points to filter
        self.fixed_points = fixed_points # bool, true if always use same points, otherwise sample

        if self.fixed_points:
            self.point_clouds=map(self.shuffle_pc, self.point_clouds)

        self.epochs_completed = 0
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

    def shuffle_pc(self, point_cloud):
        return point_cloud[np.random.permutation(point_cloud.shape[0])[:self.num_points]]

    def shuffle_data(self, seed=None):

        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = list(np.array(self.point_clouds)[perm])
        self.labels = self.labels[perm]
        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = list(np.array(self.noisy_point_clouds)[perm])
        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.

        no augementation

        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if self.noisy_point_clouds is None:
            # pdb.set_trace()
            if not self.fixed_points:
                return np.array(map(self.shuffle_pc, self.point_clouds[start:end])), self.labels[start:end], None
            else:
                return np.array(self.point_clouds[start:end]), self.labels[start:end], None
        else:
            if not self.fixed_points:
                return np.array(map(self.shuffle_pc, self.point_clouds[start:end])), self.labels[start:end], self.noisy_point_clouds[start:end, filter_points]
            else:
                return np.array(self.point_clouds[start:end]), self.labels[start:end], self.noisy_point_clouds[start:end]
    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        no augementation
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)
        pc = list(np.array(self.point_clouds)[perm])
        lb = self.labels[perm]
        ns = None
        if self.noisy_point_clouds is not None:
            ns = list(np.array(self.noisy_point_clouds)[perm])
            if not self.fixed_points:
                ns = map(self.shuffle_pc, ns)
        if not self.fixed_points:
            pc = map(self.shuffle_pc, pc)
        return np.array(pc), lb, np.array(ns)

    def merge(self, other_data_set):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.point_clouds = self.point_clouds + other_data_set.point_clouds

        labels_1 = self.labels.reshape([self.num_examples, 1])  # TODO = move to init.
        labels_2 = other_data_set.labels.reshape([other_data_set.num_examples, 1])
        self.labels = np.vstack((labels_1, labels_2))
        self.labels = np.squeeze(self.labels)

        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = self.noisy_point_clouds + other_data_set.noisy_point_clouds

        self.num_examples = len(self.point_clouds)

        return self


# vertices = load_obj('/home/yz6/data/SSE_CARS_DATA/001.obj')
# vert, vert_id = pc_loader('/home/yz6/data/SSE_CARS_DATA/001.obj')
# print(vert)

# pc1, pc2, pc3 = load_all_point_clouds_under_folder('/home/yz6/data/SSE_CARS_DATA/', file_ending='.txt', num_points=100000)
# print(pc1.next_batch(2)[0].shape)

