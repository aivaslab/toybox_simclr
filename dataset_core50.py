import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import torch
import time
import csv
import pickle
import math
from dataset_simclr import data_simclr


classes = ["plug adapters", "mobile phones", "scissors", "light bulbs", "cans", "glasses", "balls", "markers", "cups",
		   "remote controls"]

TEST_NO = [3, 7, 10]
VAL_NO = [1, 5, 9]


class data_core50(data_simclr):

	def __init__(self, root, rng, train = True, transform = None, nViews = 2, size = 224, split =
				"unsupervised", fraction = 1.0, distort = 'self', adj = -1, hyperTune = True, frac_by_object = False,
				 split_by_sess = False, distortArg = False):
		self.tr_start_key = 'Sess Start'
		self.tr_end_key = 'Sess End'
		self.obj_start_key = 'Obj Start'
		self.obj_end_key = 'Obj End'
		self.cl_start_key = 'CL Start'
		self.cl_end_key = 'CL End'
		self.tr_key = 'Session No'
		self.split_by_sess = split_by_sess
		if split_by_sess:
			if not hyperTune:
				self.trainImagesFile = "./data/core50_data_train.pickle"
				self.trainLabelsFile = "./data/core50_data_train.csv"
				self.testImagesFile = "./data/core50_data_test.pickle"
				self.testLabelsFile = "./data/core50_data_test.csv"
			else:
				self.trainImagesFile = "./data/core50_data_dev.pickle"
				self.trainLabelsFile = "./data/core50_data_dev.csv"
				self.testImagesFile = "./data/core50_data_val.pickle"
				self.testLabelsFile = "./data/core50_data_val.csv"
		else:
			if not hyperTune:
				self.trainImagesFile = "./data/core50_data_train_object.pickle"
				self.trainLabelsFile = "./data/core50_data_train_object.csv"
				self.testImagesFile = "./data/core50_data_test_object.pickle"
				self.testLabelsFile = "./data/core50_data_test_object.csv"
			else:
				self.trainImagesFile = "./data/core50_data_dev_object.pickle"
				self.trainLabelsFile = "./data/core50_data_dev_object.csv"
				self.testImagesFile = "./data/core50_data_val_object.pickle"
				self.testLabelsFile = "./data/core50_data_val_object.csv"

		super().__init__(root = root, rng = rng, train = train, transform = transform, nViews = nViews, size = size,
						 split = split, fraction = fraction, distort = distort, adj = adj, hyperTune = hyperTune,
						 frac_by_object = frac_by_object, distortArg = distortArg)

	def select_indices_object(self):
		raise NotImplementedError()


def online_mean_and_sd(loader):
	"""Compute the mean and sd in an online fashion

		Var[x] = E[X^2] - E^2[X]
	"""
	cnt = 0
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)

	for _, (data, _), _ in loader:
		print(data.shape)
		b, c, h, w = data.shape
		nb_pixels = b * h * w
		sum_ = torch.sum(data, dim=[0, 2, 3])
		sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
		fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
		snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

		cnt += nb_pixels

	return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == "__main__":
	rng = np.random.default_rng(5)
	simclr = data_core50(root = "./data", rng = rng, train = True, nViews = 2, size = 224,
								transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]), fraction = 0.1,
						 distort = "self", adj = -1, hyperTune = True, frac_by_object = True)
	trainDataLoader = torch.utils.data.DataLoader(simclr, batch_size = 64, shuffle = True,
													  num_workers = 2)

	print(len(simclr))

	# mean, std = online_mean_and_sd(trainDataLoader)
	# print(mean, std)
