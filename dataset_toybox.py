"""Module for loading the Toybox dataset"""
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

classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']

TEST_NO = {
    'ball': [1, 7, 9],
    'spoon': [5, 7, 8],
    'mug': [12, 13, 14],
    'cup': [12, 13, 15],
    'giraffe': [1, 5, 13],
    'horse': [1, 10, 15],
    'cat': [4, 9, 15],
    'duck': [5, 9, 13],
    'helicopter': [5, 10, 15],
    'airplane': [2, 6, 15],
    'truck': [2, 6, 8],
    'car': [6, 11, 13],
}

VAL_NO = {
    'airplane': [30, 29, 28],
    'ball': [30, 29, 28],
    'car': [30, 29, 28],
    'cat': [30, 29, 28],
    'cup': [30, 29, 28],
    'duck': [30, 29, 28],
    'giraffe': [30, 29, 28],
    'helicopter': [30, 29, 28],
    'horse': [30, 29, 28],
    'mug': [30, 29, 28],
    'spoon': [30, 29, 28],
    'truck': [30, 29, 28]
}


class ToyboxDataset(data_simclr):
    """Toybox dataset"""
    
    def __init__(self, root, rng, train=True, transform=None, n_views=2, size=224, split="unsupervised",
                 fraction=1.0, distort='self', adj=-1, hypertune=True, frac_by_object=False,
                 distort_arg=False, interpolate=False, umap=False):
        self.tr_start_key = 'Tr Start'
        self.tr_end_key = 'Tr End'
        self.obj_start_key = 'Obj Start'
        self.obj_end_key = 'Obj End'
        self.tr_key = 'Transformation'
        self.cl_start_key = 'CL Start'
        self.cl_end_key = 'CL End'
        self.umap = umap
        if not hypertune:
            if not interpolate:
                self.trainImagesFile = "../data/toybox_data_cropped_train.pickle"
                self.trainLabelsFile = "../data/toybox_data_cropped_train.csv"
                self.testImagesFile = "../data/toybox_data_cropped_test.pickle"
                self.testLabelsFile = "../data/toybox_data_cropped_test.csv"
            else:
                self.trainImagesFile = "../data2/toybox_data_interpolated_cropped_train.pickle"
                self.trainLabelsFile = "../data2/toybox_data_interpolated_cropped_train.csv"
                self.testImagesFile = "../data2/toybox_data_interpolated_cropped_test.pickle"
                self.testLabelsFile = "../data2/toybox_data_interpolated_cropped_test.csv"
        else:
            if not interpolate:
                self.trainImagesFile = "../data/toybox_data_cropped_dev.pickle"
                self.trainLabelsFile = "../data/toybox_data_cropped_dev.csv"
                self.testImagesFile = "../data/toybox_data_cropped_val.pickle"
                self.testLabelsFile = "../data/toybox_data_cropped_val.csv"
            else:
                self.trainImagesFile = "../data2/toybox_data_interpolated_cropped_dev.pickle"
                self.trainLabelsFile = "../data2/toybox_data_interpolated_cropped_dev.csv"
                self.testImagesFile = "../data2/toybox_data_interpolated_cropped_val.pickle"
                self.testLabelsFile = "../data2/toybox_data_interpolated_cropped_val.csv"
        
        super().__init__(root=root, rng=rng, train=train, transform=transform, nViews=n_views, size=size,
                         split=split, fraction=fraction, distort=distort, adj=adj, hyperTune=hypertune,
                         frac_by_object=frac_by_object, distortArg=distort_arg)
    
    def select_indices_object(self):
        numObjectsPerClassTrain = 27 - 3 * self.hyperTune
        numObjectsPerClassSelected = math.ceil(self.fraction * numObjectsPerClassTrain)
        objectsSelected = {}
        for cl in range(len(classes)):
            objectsInTrain = []
            for i in range(30):
                if i not in TEST_NO[classes[cl]]:
                    if self.hyperTune:
                        if i not in VAL_NO[classes[cl]]:
                            objectsInTrain.append(i)
                    else:
                        objectsInTrain.append(i)
            print(cl, objectsInTrain)
            objectsSel = self.rng.choice(objectsInTrain, numObjectsPerClassSelected)
            for obj in objectsSel:
                assert (obj not in TEST_NO[classes[cl]])
                if self.hyperTune:
                    assert (obj not in VAL_NO[classes[cl]])
            print(objectsSel)
            objectsSelected[cl] = objectsSel
        self.objectsSelected = objectsSelected
        indicesSelected = []
        with open(self.trainLabelsFile, "r") as csvFile:
            train_csvFile = list(csv.DictReader(csvFile))
        for i in range(len(train_csvFile)):
            cl, obj = train_csvFile[i]['Class ID'], train_csvFile[i]['Object']
            if int(obj) in objectsSelected[int(cl)]:
                indicesSelected.append(i)
        
        return indicesSelected


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


def get_images():
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size=224, padding=25),
                                    transforms.RandomApply([color_jitter], p=0.8),
                                    transforms.RandomGrayscale(p=0.2)])
    
    csvFile = open("./data2/toybox_data_interpolated_cropped_test.csv", "r")
    csvReader = list(csv.DictReader(csvFile))
    framesList = []
    for i in range(len(csvReader)):
        row = csvReader[i]
        if 'helicopter_10_pivothead_rxminus' in row['File Name']:
            framesList.append(i)
    with open("./data2/toybox_data_interpolated_cropped_test.pickle", "rb") as pickleFile:
        images = pickle.load(pickleFile)
    for i in range(len(framesList)):
        cv2.imwrite("../neurips images/" + "image_new_" + str(i) + ".png", cv2.imdecode(images[framesList[i]], 3))
        cv2.waitKey(0)
        img = transform(cv2.imdecode(images[framesList[i]], 3))
        img.save("../neurips images/" + "image_new_tr_" + str(i) + ".png")
    print(len(framesList))


if __name__ == "__main__":
    seed = 7
    rng = np.random.default_rng(seed)
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size=224, padding=25),
                                    transforms.RandomApply([color_jitter], p=0.8),
                                    transforms.RandomGrayscale(p=0.2), transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    
    csvFile = open("./data/toybox_data_cropped_train.csv", "r")
    csvReader = list(csv.DictReader(csvFile))
    framesList = []
    k = rng.integers(low=0, high=25000, size=1)[0]
    row = csvReader[k]
    print(row["Obj Start"], row['Obj End'], row['Tr Start'], row['Tr End'], row['CL Start'], row['CL End'])
    obj_start = row['Obj Start']
    obj_end = row['Obj End']
    tr_start = row['Tr Start']
    tr_end = row['Tr End']
    cl_start = row['CL Start']
    cl_end = row['CL End']
    
    with open("./data/toybox_data_cropped_train.pickle", "rb") as pickleFile:
        images = pickle.load(pickleFile)
    # print(len(simclr))
    original = cv2.imdecode(images[k], 3)
    cv2.imwrite("orig_orig.png", original)
    
    tr = rng.integers(low=tr_start, high=tr_end, size=1)[0]
    tra = cv2.imdecode(images[tr], 3)
    cv2.imwrite("orig_transform.png", tra)
    
    ob = rng.integers(low=obj_start, high=obj_end, size=1)[0]
    objImg = cv2.imdecode(images[ob], 3)
    cv2.imwrite("orig_obj.png", objImg)
    
    clidx = rng.integers(low=cl_start, high=cl_end, size=1)[0]
    clImg = cv2.imdecode(images[clidx], 3)
    cv2.imwrite("orig_cl.png", clImg)

# mean, std = online_mean_and_sd(trainDataLoader)
# print(mean, std)
