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


class data_simclr(torch.utils.data.Dataset):
    
    def __init__(self, root, rng, train=True, transform=None, nViews=2, size=224, split=
    "unsupervised", fraction=1.0, distort='self', adj=-1, hyperTune=True, frac_by_object=False,
                 distortArg=False):
        
        self.train = train
        self.root = root
        self.transform = transform
        self.nViews = nViews
        self.size = size
        self.split = split
        self.fraction = fraction
        self.distort = distort
        self.adj = adj
        self.hyperTune = hyperTune
        self.rng = rng
        self.objectsSelected = None
        self.distortArg = distortArg
        self.show = True
        # print(self.transform)
        
        super().__init__()
        assert (distort == 'self' or distort == 'object' or distort == 'transform' or distort == 'class')
        if self.split == "unsupervised":
            if self.train:
                with open(self.trainImagesFile, "rb") as pickleFile:
                    self.train_data = pickle.load(pickleFile)
                with open(self.trainLabelsFile, "r") as csvFile:
                    self.train_csvFile = list(csv.DictReader(csvFile))
                if frac_by_object:
                    self.indicesSelected = self.select_indices_object()
                else:
                    lenWholeData = len(self.train_data)
                    lenTrainData = int(self.fraction * lenWholeData)
                    self.indicesSelected = rng.choice(lenWholeData, lenTrainData, replace=False)
            else:
                with open(self.testImagesFile, "rb") as pickleFile:
                    self.test_data = pickle.load(pickleFile)
        else:
            if self.train:
                with open(self.trainImagesFile, "rb") as pickleFile:
                    self.train_data = pickle.load(pickleFile)
                with open(self.trainLabelsFile, "r") as csvFile:
                    self.train_csvFile = list(csv.DictReader(csvFile))
                if frac_by_object:
                    self.indicesSelected = self.select_indices_object()
                else:
                    lenWholeData = len(self.train_data)
                    lenTrainData = int(self.fraction * lenWholeData)
                    self.indicesSelected = rng.choice(lenWholeData, lenTrainData, replace=False)
            else:
                with open(self.testImagesFile, "rb") as pickleFile:
                    self.test_data = pickle.load(pickleFile)
                with open(self.testLabelsFile, "r") as csvFile:
                    self.test_csvFile = list(csv.DictReader(csvFile))
    
    def select_indices_object(self):
        raise NotImplementedError
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
            # print(cl, objectsInTrain)
            objectsSel = self.rng.choice(objectsInTrain, numObjectsPerClassSelected)
            for obj in objectsSel:
                assert (obj not in TEST_NO[classes[cl]])
                if self.hyperTune:
                    assert (obj not in VAL_NO[classes[cl]])
            # print(objectsSel)
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
    
    def __len__(self):
        if self.train:
            return len(self.indicesSelected)
        else:
            return len(self.test_data)
    
    def __getitem__(self, index):
        if self.train:
            actualIndex = self.indicesSelected[index]
            img = np.array(cv2.imdecode(self.train_data[actualIndex], 3))
            if self.split == "unsupervised":
                label = self.train_csvFile[actualIndex]['Class ID']
            else:
                label = self.train_csvFile[actualIndex]['Class ID']
        else:
            actualIndex = index
            img = np.array(cv2.imdecode(self.test_data[index], 3))
            label = self.test_csvFile[index]['Class ID']
        
        if self.split == "unsupervised":
            if self.distort == 'self':
                if self.transform is not None:
                    imgs = [self.transform(img) for _ in range(self.nViews)]
                else:
                    imgs = [img, img]
            
            elif self.distort == 'object':
                low, high = int(self.train_csvFile[actualIndex][self.obj_start_key]), \
                            int(self.train_csvFile[actualIndex][self.obj_end_key])
                id2 = self.rng.integers(low=low, high=high + 1, size=1)[0]
                img2 = np.array(cv2.imdecode(self.train_data[id2], 3))
                if self.transform is not None:
                    imgs = [self.transform(img), self.transform(img2)]
                else:
                    imgs = [img, img2]
            elif self.distort == 'class':
                low, high = int(self.train_csvFile[actualIndex][self.cl_start_key]), \
                            int(self.train_csvFile[actualIndex][self.cl_end_key])
                id2 = self.rng.integers(low=low, high=high + 1, size=1)[0]
                img2 = np.array(cv2.imdecode(self.train_data[id2], 3))
                if self.transform is not None:
                    imgs = [self.transform(img), self.transform(img2)]
                else:
                    imgs = [img, img2]
            else:
                if self.adj == -1:
                    low, high = int(self.train_csvFile[actualIndex][self.tr_start_key]), int(
                        self.train_csvFile[actualIndex][self.tr_end_key])
                    id2 = self.rng.integers(low=low, high=high + 1, size=1)[0]
                else:
                    low = max(0, actualIndex - self.adj)
                    high = min(int(len(self.train_data)) - 1, actualIndex + self.adj)
                    try:
                        if self.train_csvFile[low][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
                            id2 = high
                        elif self.train_csvFile[high][self.tr_key] != self.train_csvFile[actualIndex][self.tr_key]:
                            id2 = low
                        else:
                            if self.distortArg:
                                id2 = self.rng.choice([low, high], 1)[0]
                            else:
                                id2 = self.rng.integers(low=low, high=high + 1, size=1)[0]
                    except IndexError:
                        print(low, actualIndex, high)
                img2 = np.array(cv2.imdecode(self.train_data[id2], 3))
                if self.transform is not None:
                    imgs = [self.transform(img), self.transform(img2)]
                else:
                    imgs = [img, img2]
        else:
            if self.transform is not None:
                imgs = self.transform(img)
            else:
                imgs = img
        if self.show:
            imgl = transforms.ToPILImage()(imgs[0])
            if self.show:
                # imgl.show()
                self.show = False
        if not self.umap:
            return actualIndex, imgs, int(label)
        else:
            return index, imgs, int(label)
