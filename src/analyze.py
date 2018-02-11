import numpy as np
import sys
from collections import defaultdict
import pprint
from latent_3d_points.external.structural_losses.tf_nndistance import nn_distance
import pdb
import tensorflow as tf


def get_chamfer_permut(data, permut):
	assert len(permut) == len(data)
	permut = np.array(permut)
	c1, _, c2, _ = nn_distance(data, data[permut])
	sess = tf.Session()
	c1, c2 = sess.run([c1, c2])
	dist = np.mean(c1) + np.mean(c2)
	return dist


def get_nn_chamfer_own(data):
	l_data = len(data)
	l_fake = l_data
	data_fake = data
	sess = tf.Session()
	dist_list = list()
	fake_to_true = defaultdict(list)
	true_to_fake = defaultdict(list)
	for j in range(l_fake):
		fake_data = np.tile(data_fake[j], (l_data-1, 1, 1))
		c1, _, c2, _ = nn_distance(fake_data, np.concatenate((data[:j], data[j+1:]), axis=0))
		c1, c2 = sess.run([c1, c2])
		dist = np.mean(c1 , axis=1) + np.mean(c2, axis=1)
		code = dist.argmin()
		dist = dist.min()
		fake_to_true[j] = (code, dist)
		true_to_fake[code].append((j, dist))
		dist_list.append(dist)	
		print('compared to %s: %s' %(j, np.mean(dist_list)))
	return fake_to_true, true_to_fake, dist_list

def get_avg_chamfer_own(data):
	l_data = len(data)
	l_fake = l_data
	data_fake = data
	sess = tf.Session()
	dist_list = list()
	for j in range(l_fake):
		fake_data = np.tile(data_fake[j], (l_data-1, 1, 1))
		c1, _, c2, _ = nn_distance(fake_data, np.concatenate((data[:j], data[j+1:]), axis=0))
		c1, c2 = sess.run([c1, c2])
		dist = np.mean(c1 , axis=1) + np.mean(c2, axis=1)
		dist = dist.mean()
		dist_list.append(dist)	
		print('compared to %s: %s' %(j, np.mean(dist_list)))
	return fake_to_true, true_to_fake, dist_list



def get_nn_chamfer(data, data_fake):
	l_data = len(data)
	l_fake = len(data_fake)	
	sess = tf.Session()
	dist_list = list()
	fake_to_true = defaultdict(list)
	true_to_fake = defaultdict(list)
	for j in range(l_fake):
		fake_data = np.tile(data_fake[j], (l_data, 1, 1))
		c1, _, c2, _ = nn_distance(fake_data, data)
		c1, c2 = sess.run([c1, c2])
		dist = np.mean(c1 , axis=1) + np.mean(c2, axis=1)
		code = dist.argmin()
		dist = dist.min()
		fake_to_true[j] = (code, dist)
		true_to_fake[code].append((j, dist))
		dist_list.append(dist)	
		print('compared to %s: %s' %(j, np.mean(dist_list)))
	return fake_to_true, true_to_fake, dist_list



def get_mean_and_std_of_latent_code_ae(data):
	num_samples, hidden_dim = data.shape

	mean = data.mean(axis=0)
	assert len(mean) == hidden_dim, "mean does not match hidden dimension"

	std = data.std(axis=0)
	assert len(std) == hidden_dim, "std does not match hidden dimension"

	min_data, max_data = data.min(), data.max()

	print("mean")
	print(repr(mean))

	print("std")
	print(repr(std))

	print("mean-min, mean-max")
	print(mean.min(), mean.max())

	print("std-min, std-max")
	print(std.min(), std.max())

	print("data-min, data-max")
	print(min_data, max_data)



	avg_dist = 0
	for i in range(len(data)):
		for j in range(len(data)):
			if i == j:
				pass
			avg_dist += np.sqrt(((data[j] - data[i]) * (data[j] - data[i])).sum())

	avg_dist /= (len(data) * (len(data) -  1))

	print("avg dist")
	print(avg_dist)

	print("std sum")
	print(np.sqrt(std*std).sum())



	return mean, std, min_data, max_data


def get_codes_with_min_diff(data):

	min_diff = np.inf
	c1, c2 = None, None
	num_samples = len(data)
	for i in range(num_samples):
		for j in range(i+1, num_samples):
			diff = (data[i] - data[j])
			diff *= diff
			diff = diff.sum()
			if diff < min_diff:
				min_diff = diff
				c1, c2 = data[i], data[j]

	print("c1")
	print(repr(c1))
	print("c2")
	print(repr(c2))

	return c1, c2



def get_codes_with_max_diff(data):
	max_diff = 0
	c1, c2 = None, None
	num_samples = len(data)
	for i in range(num_samples):
		for j in range(i+1, num_samples):
			diff = (data[i] - data[j])
			diff *= diff
			diff = diff.sum()
			if diff > max_diff:
				max_diff = diff
				c1, c2 = data[i], data[j]

	print("c1")
	print(repr(c1))
	print("c2")
	print(repr(c2))

	return c1, c2
def get_average_distance(code):
	num_samples = len(code)
	total_dist = 0 
	for i in range(num_samples):
		for j in range(num_samples):
			if i == j:
				continue
			dist = (code[i] - code[j]) * (code[i] - code[j])
			dist = dist.sum()
			total_dist += np.sqrt(dist)
	total_dist /= num_samples*(num_samples - 1)
	return total_dist

def get_nn_distance(code):
	num_samples = len(code)
	average_nn_dist = 0 
	nearest_list = list()
	for i in range(num_samples):
		min_dist = np.inf
		nn_index = 0
		for j in range(num_samples):
			if i == j:
				continue
			dist = (code[i] - code[j]) * (code[i] - code[j])
			dist = np.sqrt(dist.sum())
			if dist < min_dist:
				min_dist = min(min_dist, dist)
				nn_index = j
		average_nn_dist += min_dist
		nearest_list.append(nn_index)
	average_nn_dist /= num_samples
	return nearest_list, average_nn_dist

def get_nn_between_codes(code, gen_code):
	# pdb.set_trace()
	dist_list = list()
	code_to_gen = defaultdict(list)
	gen_to_code = defaultdict(list)
	l = len(code)
	l_gen = len(gen_code)
	for j in range(l_gen):
		min_dist= np.inf
		code_index=0
		for i in range(l):
			dist = gen_code[j] - code[i]
			dist *=dist
			dist = np.sqrt(dist.sum())
			if dist < min_dist:
				min_dist = dist
				code_index = i
		gen_to_code[j] = (code_index, min_dist)
		code_to_gen[code_index].append((j, min_dist))
		dist_list.append(min_dist)

	return code_to_gen, gen_to_code,  dist_list, np.mean(dist_list)


# if __name__=="__main__":

# 	file_hidden_code = sys.argv[1] # path to hidden code file
# 	file_hidden_code_gen = sys.argv[2] # path to geenrated hidden code file


# 	data = np.load(file_hidden_code)
# 	data_gen = np.load(file_hidden_code_gen)

# 	mean, std, min_data, max_data = get_mean_and_std_of_latent_code_ae(data)

# 	# c1, c2 = get_codes_with_max_diff(data)

# 	d, distances = get_nn(data, data_gen)

# 	# for k, v in d.items():
# 	# 	print(k)
# 	# 	print(v)
# 	# print(len(d))

# 	print("mean of distance")
# 	print(np.sqrt(np.array(distances)).mean())






