

import numpy as np


class DataLoader(object):
    def __init__(self, file, batch_size, shuffle=True, repeat=True):
        self.data = np.load(file) #[:, 128]
        self.len_data = len(self.data) 
        self.shuffle = shuffle
        self.repeat = repeat
        self.batch_size = batch_size


    def __iter__(self):
        return self.iterator()

    def iterator(self):
        idx_perm = np.arange(self.len_data)

        while(True):

            if self.shuffle:
                idx_perm = np.random.permutation(self.len_data)
            for i in range(0, self.len_data, self.batch_size):
                if self.len_data - i < self.batch_size:
                    break
                idx_start = i
                idx_end = min(i + self.batch_size, self.len_data)
                idx_range = idx_perm[idx_start : idx_end]
                yield self.data[idx_range]
            if self.repeat:
                continue
            else:
                break