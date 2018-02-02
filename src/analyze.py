import numpy as np
import sys
from collections import defaultdict
import pprint
import pdb

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


	# nn_diff = 0
	# for i in range(len(data)):
	# 	nn_dist = np.inf 
	# 	for j in range(len(data)):
	# 		if i == j:
	# 			continue
	# 		dist = data[j] - data[i]
	# 		dist *= dist
	# 		dist = dist.sum()
	# 		dist = np.sqrt(dist)
	# 		nn_dist = min(nn_dist, dist)
	# 	nn_diff +=  nn_dist

	# print("mean nn distances")
	# print(nn_diff/ len(data))


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

def get_nn(code, gen_code):
	# pdb.set_trace()
	dist_list = list()
	d = defaultdict(list)
	l = len(code)
	l_gen = len(gen_code)
	for j in range(l_gen):
		min_dist= np.inf
		code_index=0
		for i in range(l):
			dist = gen_code[j] - code[i]
			dist *=dist
			dist = dist.sum()
			if dist < min_dist:
				min_dist = dist
				code_index = i
		d[code_index].append((j, min_dist))
		dist_list.append(min_dist)
	return d, dist_list


if __name__=="__main__":

	file_hidden_code = sys.argv[1] # path to hidden code file
	file_hidden_code_gen = sys.argv[2] # path to geenrated hidden code file


	data = np.load(file_hidden_code)
	data_gen = np.load(file_hidden_code_gen)

	mean, std, min_data, max_data = get_mean_and_std_of_latent_code_ae(data)

	# c1, c2 = get_codes_with_max_diff(data)

	d, distances = get_nn(data, data_gen)

	# for k, v in d.items():
	# 	print(k)
	# 	print(v)
	# print(len(d))

	print("mean of distance")
	print(np.sqrt(np.array(distances)).mean())






