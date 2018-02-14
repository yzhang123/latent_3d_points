import numpy as np 
import os.path as osp
import re
import os
import pymesh
import csv
import sys
from collections import defaultdict
import pdb



r_min, r_max = 3, 6
batch_size = 64
cx_min, cx_max, cy_min, cy_max = 8, 8, 8, 8



def generate_circles(batch_size, num_points=2048):
    """
    return train, val and test instances of PointCloudDataSet
    """
    radius = np.random.uniform(r_min, r_max, [batch_size, 1, 1])
    center_x = np.random.randn(batch_size, 1, 1)*cx_max*0.5
    center_y = np.random.randn(batch_size, 1, 1)*cx_max*0.5
    quadrant = np.random.randint(0, 4, [batch_size, 1, 1])

    offset = 2*cx_max
    add_x = np.array([offset,  offset, -offset, -offset])
    add_y = np.array([offset, -offset, -offset,  offset])
    center_x += add_x[quadrant]
    center_y += add_y[quadrant]

    angle = 2 * np.pi * np.random.uniform(0, 1, [batch_size, num_points, 1])

    points_x = radius * np.cos(angle) + center_x
    points_y = radius * np.sin(angle) + center_y

    points = np.concatenate([points_x, points_y], axis=2).astype(np.float32)  # batch_size x num_points x 2

    return points



class SyntheticPointCloudDataSet(object):
    def __init__(self, num_examples=5000, num_points=2048, labels=None):
        '''Construct a DataSet.
        Args:
            init_shuffle, shuffle data before first epoch has been reached.
        Output:
            original_pclouds, labels, (None or Feed) # TODO Rename
        '''
        self.num_examples = num_examples
        
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        self.num_points = num_points # number of points to filter
        self.epochs_completed = 0
        self._index_in_epoch = 0


    def shuffle_data(self, seed=None):
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

        return generate_circles(batch_size, num_points=self.num_points), None, None

    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        no augementation
        '''
        return generate_circles(batch_size=self.num_examples, num_points=self.num_points), self.labels, np.array(None)


