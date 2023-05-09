import torch
import torch.nn.functional as F
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class TripletMarginWithDistanceLoss:
    def __init__(self, distanceFunction = None, margin = 0.5):
        self.distanceFunction = distanceFunction
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        if self.distanceFunction is None:
            positive_dist = torch.pairwise_distance(anchor, positive)
            negative_dist = torch.pairwise_distance(anchor, negative)
        else:
            positive_dist = self.distanceFunction(anchor, positive)
            negative_dist = self.distanceFunction(anchor, negative)
        # print(positive_dist.shape, negative_dist.shape)
        return torch.clamp(positive_dist - negative_dist + self.margin, min = 0.0).mean()


def triple_info_nce_loss(features, temp):
    dev = torch.device('cuda:0')
    batchSize = features.shape[0] / 3
    labels = torch.cat([torch.arange(batchSize) for _ in range(3)], dim = 0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dev)

    features = F.normalize(features, dim = 1)

    similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype = torch.uint8).to(dev)
    labels = labels[~mask].view(labels.shape[0], -1).type(torch.uint8)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim = 1)
    labels1 = torch.zeros(logits.shape[0], dtype = torch.long).to(dev)
    labels2 = torch.ones(logits.shape[0], dtype = torch.long).to(dev)

    logits = logits / temp
    return logits, labels1, labels2


def info_nce_loss(features, temp):
    dev = torch.device('cuda:0')
    batchSize = features.shape[0] / 2
    labels = torch.cat([torch.arange(batchSize) for _ in range(2)], dim = 0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dev)
    features = F.normalize(features, dim = 1)
    similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype = torch.bool).to(dev)
    labels = labels[~mask].view(labels.shape[0], -1).type(torch.uint8)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim = 1)
    labels = torch.zeros(logits.shape[0], dtype = torch.long).to(dev)
    logits = logits / temp
    # raise NotImplementedError()
    return logits, labels


def calc_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batchSize = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target_reshaped = torch.reshape(target, (1, -1)).repeat(maxk, 1)
        correct_top_k = torch.eq(pred, target_reshaped)
        pred_1 = pred[0]
        res = []
        for k in topk:
            correct_k = correct_top_k[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(torch.mul(correct_k, 100.0 / batchSize))
        return res, pred_1


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
