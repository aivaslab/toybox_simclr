"""Module which holds datasets for transfer learning"""
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
import csv
import cv2
import torch.utils.data


class fCIFAR10(CIFAR10):
    """Fractional cifar dataset"""
    def __init__(self, root, train, download, transform, rng, hypertune=True, frac=1.0):
        if hypertune:
            super(fCIFAR10, self).__init__(root=root, train=True, download=download)
        else:
            super(fCIFAR10, self).__init__(root=root, train=train, download=download)
        self.train = train
        self.transform = transform
        self.frac = frac
        self.hypertune = hypertune
        self.rng = rng
        
        if self.hypertune:
            if self.train:
                range_low = 0
                range_high = int(0.8 * len(self.data))
            else:
                range_low = int(0.8 * len(self.data))
                range_high = len(self.data)
        else:
            range_low = 0
            range_high = len(self.data)
        
        arr = np.arange(range_low, range_high)
        print("Split:", self.train, np.min(arr), np.max(arr))
        len_data = range_high - range_low
        
        indices = self.rng.choice(arr, size=int(frac * len_data), replace=False)
        
        unique = len(indices) == len(set(indices))
        assert unique
        assert len(indices) == int(frac * len_data)
        
        if self.train:
            self.train_data = self.data[indices]
            self.train_labels = np.array(self.targets)[indices]
        else:
            self.test_data = self.data[indices]
            self.test_labels = np.array(self.targets)[indices]
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_labels)
    
    def __getitem__(self, item):
        if self.train:
            img = self.train_data[item]
            target = self.train_labels[item]
        else:
            img = self.test_data[item]
            target = self.test_labels[item]
        img = self.transform(img)
        return item, img, target


class fCIFAR100(CIFAR100):
    """Fractional Cifar100"""
    def __init__(self, root, train, download, transform, rng, hypertune=True, frac=1.0):
        
        if hypertune:
            super(fCIFAR100, self).__init__(root=root, train=True, download=download)
        else:
            super(fCIFAR100, self).__init__(root=root, train=train, download=download)
        self.train = train
        self.transform = transform
        self.frac = frac
        self.hypertune = hypertune
        self.rng = rng
        
        if self.hypertune:
            if self.train:
                range_low = 0
                range_high = int(0.8 * len(self.data))
            else:
                range_low = int(0.8 * len(self.data))
                range_high = len(self.data)
        else:
            range_low = 0
            range_high = len(self.data)
        
        arr = np.arange(range_low, range_high)
        print("Split:", self.train, np.min(arr), np.max(arr))
        len_data = range_high - range_low
        
        indices = self.rng.choice(arr, size=int(frac * len_data), replace=False)
        
        unique = len(indices) == len(set(indices))
        assert unique
        assert len(indices) == int(frac * len_data)
        
        if self.train:
            self.train_data = self.data[indices]
            self.train_labels = np.array(self.targets)[indices]
        else:
            self.test_data = self.data[indices]
            self.test_labels = np.array(self.targets)[indices]
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_labels)
    
    def __getitem__(self, item):
        if self.train:
            img = self.train_data[item]
            target = self.train_labels[item]
        else:
            img = self.test_data[item]
            target = self.test_labels[item]
        img = self.transform(img)
        return item, img, target


class fCORe50(torch.utils.data.Dataset):
    """Fractional Core50"""
    def __init__(self, train, transform, rng, hypertune=True, frac=1.0):
        
        if hypertune:
            super(fCORe50, self).__init__()
        else:
            super(fCORe50, self).__init__()
        self.train = train
        self.transform = transform
        self.frac = frac
        self.hypertune = hypertune
        self.rng = rng
        
        if not hypertune:
            self.trainImagesFile = "../data/core50_data_train_object.pickle"
            self.trainLabelsFile = "../data/core50_data_train_object.csv"
            self.testImagesFile = "../data/core50_data_test_object.pickle"
            self.testLabelsFile = "../data/core50_data_test_object.csv"
        else:
            self.trainImagesFile = "../data/core50_data_dev_object.pickle"
            self.trainLabelsFile = "../data/core50_data_dev_object.csv"
            self.testImagesFile = "../data/core50_data_val_object.pickle"
            self.testLabelsFile = "../data/core50_data_val_object.csv"
        
        if self.train:
            with open(self.trainImagesFile, "rb") as pickleFile:
                self.data = pickle.load(pickleFile)
            with open(self.trainLabelsFile, "r") as csvFile:
                self.labels = list(csv.DictReader(csvFile))
        else:
            with open(self.testImagesFile, "rb") as pickleFile:
                self.data = pickle.load(pickleFile)
            with open(self.testLabelsFile, "r") as csvFile:
                self.labels = list(csv.DictReader(csvFile))
        
        len_data = len(self.data)
        self.indices = rng.choice(len_data, size=int(frac * len_data), replace=False)
    
    def __len__(self):
        if self.train:
            return len(self.indices)
        else:
            return len(self.indices)
    
    def __getitem__(self, idx):
        item = self.indices[idx]
        img = np.array(cv2.imdecode(self.data[item], 3))
        target = int(self.labels[item]['Class ID'])
        img = self.transform(img)
        return item, img, target


class fALOI(torch.utils.data.Dataset):
    """Fractional ALOI dataset"""
    def __init__(self, train, transform, rng, hypertune=True, frac=1.0):
        if hypertune:
            super(fALOI, self).__init__()
        else:
            super(fALOI, self).__init__()
        self.train = train
        self.transform = transform
        self.frac = frac
        self.hypertune = hypertune
        self.rng = rng
        
        if not hypertune:
            self.trainImagesFile = "../data/aloi_train.pickle"
            self.trainLabelsFile = "../data/aloi_train.csv"
            self.testImagesFile = "../data/aloi_test.pickle"
            self.testLabelsFile = "../data/aloi_test.csv"
        else:
            self.trainImagesFile = "../data/aloi_dev.pickle"
            self.trainLabelsFile = "../data/aloi_dev.csv"
            self.testImagesFile = "../data/aloi_val.pickle"
            self.testLabelsFile = "../data/aloi_val.csv"
        
        if self.train:
            with open(self.trainImagesFile, "rb") as pickleFile:
                self.data = pickle.load(pickleFile)
            with open(self.trainLabelsFile, "r") as csvFile:
                self.labels = list(csv.DictReader(csvFile))
        else:
            with open(self.testImagesFile, "rb") as pickleFile:
                self.data = pickle.load(pickleFile)
            with open(self.testLabelsFile, "r") as csvFile:
                self.labels = list(csv.DictReader(csvFile))
        assert (len(self.data) == len(self.labels))
        len_data = len(self.data)
        self.indices = rng.choice(len_data, size=int(frac * len_data), replace=False)
    
    def __len__(self):
        if self.train:
            return len(self.indices)
        else:
            return len(self.indices)
    
    def __getitem__(self, idx):
        item = self.indices[idx]
        img = np.array(cv2.imdecode(self.data[item], 3))
        target = int(self.labels[item]['Object ID']) - 1
        img = self.transform(img)
        return item, img, target


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion
    Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    
    for _, data, _ in loader:
        # print(data.shape)
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        
        cnt += nb_pixels
    
    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


def main():
    """Main method"""
    transform = transforms.Compose([transforms.ToPILImage(),
                                    # transforms.Resize(224),
                                    transforms.ToTensor(),
                                    ])
    dataset = fALOI(train=False, transform=transform, hypertune=False,
                    frac=1.0, rng=np.random.default_rng(0))
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    m, sd = online_mean_and_sd(loader=loader)
    print(m, sd)
    cv2.imshow("example", cv2.imdecode(dataset[0][1], 3))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
