"""Module to run the transfer learner experiments starting from simclr backbones"""
import numpy as np
import torchvision.transforms as transforms
import torch
import network as simclr_net
import argparse
import os
import torch.nn as nn
import utils
import tqdm
import cifar10_frac
import torch.utils.data
import torch.backends.cudnn

import dataset_imagenet12

cifar10_mean = (0.4920, 0.4827, 0.4468)
cifar10_std = (0.2471, 0.2435, 0.2617)
cifar100_mean = (0.5053, 0.4863, 0.4415)
cifar100_std = (0.2674, 0.2563, 0.2763)
core50_mean = (0.6000, 0.5691, 0.5418)
core50_std = (0.2137, 0.2211, 0.2352)
aloi_mean = (0.0638, 0.0988, 0.1158)
aloi_std = (0.1272, 0.1681, 0.1964)
IN12_MEAN = (0.4541, 0.4845, 0.4980)
IN12_STD = (0.2928, 0.2738, 0.2756)
IN12_DATA_PATH = "../../toybox_journal_experiments/data_12/IN-12/"


def get_parser(desc):
    """Create and return the parser"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', '-m', required=True, type=str)
    parser.add_argument('--lr', '-lr', required=True, type=float)
    parser.add_argument('--epochs', '-e', required=True, type=int)
    parser.add_argument('--fraction', '-f', default=1.0, type=float)
    parser.add_argument('--dataset', '-data', default="cifar10", type=str)
    parser.add_argument('--combined', '-c', default=False, action="store_true")
    parser.add_argument('--batch-size', '-b', default=128, type=int)
    parser.add_argument('--num-reps', '-n', default=3, type=int)
    parser.add_argument('--hypertune', '-ht', default=False, action='store_true')
    parser.add_argument('--num_layers_frozen', '-l', default=4, type=int)
    parser.add_argument('--decay_epochs', '-de', default=20, type=int)
    
    return parser.parse_args()


def set_seed(sd):
    """Set seed for experiments"""
    if sd == -1:
        sd = np.random.randint(0, 65536)
    print("Setting seed to", sd)
    torch.manual_seed(sd)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(sd)
    return rng


def run_transfer_learner(args):
    """run the transfer learning experiment"""
    network = simclr_net.SimClRNet(num_classes=12).cuda()
    
    if args['model'] != "random":
        fileName = args['model']
        assert (os.path.isfile(fileName))
        network.load_state_dict(torch.load(fileName))
    else:
        print("Random model initialized.")
    
    network.unfreeze_all_params()
    totalParams = sum(p.numel() for p in network.backbone.parameters())
    trainableParams = sum(p.numel() for p in network.backbone.parameters() if p.requires_grad)
    print(str(trainableParams) + "/" + str(totalParams) + " parameters of backbone network are trainable.")
    
    if args['num_layers_frozen'] == 1:
        network.freeze_backbone_layer_1()
        totalParams = sum(p.numel() for p in network.backbone.parameters())
        trainableParams = sum(p.numel() for p in network.backbone.parameters() if p.requires_grad)
        print(str(trainableParams) + "/" + str(totalParams) + " parameters of backbone network are trainable.")
    elif args['num_layers_frozen'] == 2:
        network.freeze_backbone_layer_2()
        totalParams = sum(p.numel() for p in network.backbone.parameters())
        trainableParams = sum(p.numel() for p in network.backbone.parameters() if p.requires_grad)
        print(str(trainableParams) + "/" + str(totalParams) + " parameters of backbone network are trainable.")
    elif args['num_layers_frozen'] == 3:
        network.freeze_backbone_layer_3()
        totalParams = sum(p.numel() for p in network.backbone.parameters())
        trainableParams = sum(p.numel() for p in network.backbone.parameters() if p.requires_grad)
        print(str(trainableParams) + "/" + str(totalParams) + " parameters of backbone network are trainable.")
    elif args['num_layers_frozen'] > 0:
        network.freeze_all_params()
        totalParams = sum(p.numel() for p in network.backbone.parameters())
        trainableParams = sum(p.numel() for p in network.backbone.parameters() if p.requires_grad)
        print(str(trainableParams) + "/" + str(totalParams) + " parameters of backbone network are trainable.")
    
    featSize = network.classifier_fc.in_features
    
    if args['dataset'] == "cifar10":
        trnsfr_classifier = nn.Sequential(nn.Linear(featSize, featSize // 2), nn.ReLU(),
                                          nn.Linear(featSize // 2, 10)).cuda()
        trainTransform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomCrop(size=224, padding=5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(cifar10_mean, cifar10_std)])
        testTransform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cifar10_mean, cifar10_std)
                                            ])
        
        trainData = cifar10_frac.fCIFAR10(root="../data", train=True, transform=trainTransform, download=True,
                                          frac=args['fraction'], hypertune=args['hypertune'], rng=args['rng'])
        data_test = cifar10_frac.fCIFAR10(root="../data", train=False, download=True,
                                          transform=testTransform, hypertune=args['hypertune'], rng=args['rng'])
    elif args['dataset'] == "cifar100":
        trnsfr_classifier = nn.Sequential(nn.Linear(featSize, featSize // 2), nn.ReLU(),
                                          nn.Linear(featSize // 2, 100)).cuda()
        trainTransform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomCrop(size=224, padding=5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(cifar100_mean, cifar100_std)])
        
        testTransform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cifar100_mean, cifar100_std)
                                            ])
        trainData = cifar10_frac.fCIFAR100(root="../data", train=True, transform=trainTransform, download=True,
                                           frac=args['fraction'], hypertune=args['hypertune'], rng=args['rng'])
        data_test = cifar10_frac.fCIFAR100(root="../data", train=False, download=True,
                                           transform=testTransform, hypertune=args['hypertune'], rng=args['rng'])
    elif args["dataset"] == "core50":
        trnsfr_classifier = nn.Sequential(nn.Linear(featSize, featSize // 2), nn.ReLU(),
                                          nn.Linear(featSize // 2, 10)).cuda()
        trainTransform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomCrop(size=224, padding=5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(core50_mean, core50_std)])
        
        testTransform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(core50_mean, core50_std)
                                            ])
        trainData = cifar10_frac.fCORe50(train=True, transform=trainTransform, frac=args['fraction'],
                                         hypertune=args['hypertune'], rng=args['rng'])
        data_test = cifar10_frac.fCORe50(train=False, transform=testTransform, hypertune=args['hypertune'],
                                         rng=args['rng'])
    elif args["dataset"] == "in12":
        trnsfr_classifier = nn.Sequential(nn.Linear(featSize, featSize // 2), nn.ReLU(),
                                          nn.Linear(featSize // 2, 12)).cuda()
        trainTransform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomCrop(size=224, padding=5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)])
    
        testTransform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=IN12_MEAN, std=IN12_STD)
                                            ])
        trainData = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=True, transform=trainTransform,
                                                         fraction=args['fraction'], hypertune=args['hypertune'])

        data_test = dataset_imagenet12.DataLoaderGeneric(root=IN12_DATA_PATH, train=False, transform=trainTransform,
                                                         fraction=args['fraction'], hypertune=args['hypertune'])
    else:
        trnsfr_classifier = nn.Sequential(nn.Linear(featSize, featSize // 2), nn.ReLU(),
                                          nn.Linear(featSize // 2, 1000)).cuda()
        trainTransform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomCrop(size=224, padding=5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(aloi_mean, aloi_std)])
        
        testTransform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(aloi_mean, aloi_std)
                                            ])
        trainData = cifar10_frac.fALOI(train=True, transform=trainTransform, frac=args['fraction'],
                                       hypertune=args['hypertune'], rng=args['rng'])
        data_test = cifar10_frac.fALOI(train=False, transform=testTransform, hypertune=args['hypertune'],
                                       rng=args['rng'])
    
    trnsfr_classifier.apply(utils.init_weights)
    totalParams = sum(p.numel() for p in trnsfr_classifier.parameters())
    trainableParams = sum(p.numel() for p in trnsfr_classifier.parameters() if p.requires_grad)
    print(str(trainableParams) + "/" + str(totalParams) + " parameters of transfer classifier are trainable.")
    print("Train data size:", len(trainData))
    print("Test data size:", len(data_test))
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=args['batch_size'], shuffle=True,
                                                  num_workers=4)
    
    testDataLoader = torch.utils.data.DataLoader(data_test, batch_size=args['batch_size'], shuffle=True,
                                                 num_workers=4)
    if True:
        print("train data: ", len(trainData), utils.online_mean_and_sd(trainDataLoader))
        print("train data: ", len(data_test), utils.online_mean_and_sd(testDataLoader))
        
    optimizer = torch.optim.SGD(trnsfr_classifier.parameters(), lr=args['lr'], weight_decay=1e-6, momentum=0.9)
    if args['num_layers_frozen'] <= 3:
        optimizer.add_param_group({'params': network.backbone.layer4.parameters()})
        optimizer.add_param_group({'params': network.backbone.avgpool.parameters()})
    if args['num_layers_frozen'] <= 2:
        optimizer.add_param_group({'params': network.backbone.layer3.parameters()})
    if args['num_layers_frozen'] <= 1:
        optimizer.add_param_group({'params': network.backbone.layer2.parameters()})
    if args['num_layers_frozen'] <= 0:
        optimizer.add_param_group({'params': network.backbone.layer1.parameters()})
        optimizer.add_param_group({'params': network.backbone.maxpool.parameters()})
        optimizer.add_param_group({'params': network.backbone.relu.parameters()})
        optimizer.add_param_group({'params': network.backbone.bn1.parameters()})
        optimizer.add_param_group({'params': network.backbone.conv1.parameters()})
    
    # optimizer.add_param_group({'params': network.backbone.parameters()})
    
    numEpochs = args['epochs']
    for ep in range(numEpochs):
        avg_loss = 0
        b = 0
        tqdmBar = tqdm.tqdm(trainDataLoader)
        for idx, (_, images, targets) in enumerate(tqdmBar):
            optimizer.zero_grad()
            b += 1
            images = images.cuda()
            targets = targets.to(torch.device('cuda:0'))
            feats = network.backbone(images)
            logits = trnsfr_classifier(feats)
            loss = nn.CrossEntropyLoss()(logits, targets)
            avg_loss = (avg_loss * (b - 1) + loss.item()) / b
            loss.backward()
            optimizer.step()
            tqdmBar.set_description("Epoch: {:d}/{:d} Loss: {:.4f}, LR: {:.8f}".format(ep + 1, numEpochs, avg_loss,
                                                                                       optimizer.param_groups[0]['lr']))
        if ep % args['decay_epochs'] == args['decay_epochs'] - 1 and ep > 0:
            optimizer.param_groups[0]['lr'] *= 0.7
            top1acc = 0
            totTrainPoints = 0
            for idx, (_, images, labels) in enumerate(trainDataLoader):
                with torch.no_grad():
                    feats = network.backbone(images.cuda())
                    logits = trnsfr_classifier(feats)
                top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1,))
                top1acc += top[0].item() * pred.shape[0]
                totTrainPoints += pred.shape[0]
            top1acc /= totTrainPoints
            
            print("Train Accuracies 1:", top1acc)
            
            top1acc = 0
            top2acc = 0
            totTestPoints = 0
            for _, (_, images, labels) in enumerate(testDataLoader):
                with torch.no_grad():
                    feats = network.backbone(images.cuda())
                    logits = trnsfr_classifier(feats)
                top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1, 5))
                top1acc += top[0].item() * pred.shape[0]
                top2acc += top[1].item() * pred.shape[0]
                totTestPoints += pred.shape[0]
            top1acc /= totTestPoints
            top2acc /= totTestPoints
            
            print("Test Accuracies 1 and 5:", top1acc, top2acc)
    
    top1acc_train = 0
    totTrainPoints = 0
    for idx, (_, images, labels) in enumerate(trainDataLoader):
        with torch.no_grad():
            feats = network.backbone(images.cuda())
            logits = trnsfr_classifier(feats)
        top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1,))
        top1acc_train += top[0].item() * pred.shape[0]
        totTrainPoints += pred.shape[0]
    top1acc_train /= totTrainPoints
    
    print("Train Accuracies 1:", top1acc_train)
    
    top1acc_test = 0
    top2acc = 0
    totTestPoints = 0
    for _, (_, images, labels) in enumerate(testDataLoader):
        with torch.no_grad():
            feats = network.backbone(images.cuda())
            logits = trnsfr_classifier(feats)
        top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1, 5))
        top1acc_test += top[0].item() * pred.shape[0]
        top2acc += top[1].item() * pred.shape[0]
        totTestPoints += pred.shape[0]
    top1acc_test /= totTestPoints
    top2acc /= totTestPoints
    
    print("Test Accuracies 1 and 5:", top1acc_test, top2acc)
    
    return top1acc_train, top1acc_test


def run_transfer_learner_reps(exp_args):
    """Run transfer learner repetitions"""
    train_accs = []
    test_accs = []
    for i in range(exp_args["num_reps"]):
        print("----------------------------------------------------------------------------------------")
        print("Starting run", i + 1, "of", exp_args["num_reps"])
        exp_args["rng"] = set_seed(sd=-1)
        train_acc, test_acc = run_transfer_learner(args=exp_args)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print("----------------------------------------------------------------------------------------")
    print("Aggregate after running", exp_args['num_reps'], "repetitions:")
    print("Train accuracy:", train_accs, "mean:", np.mean(np.asarray(train_accs)), "std:",
          np.std(np.asarray(train_accs)))
    print("Test accuracy:", test_accs, "mean:", np.mean(np.asarray(test_accs)), "std:", np.std(np.asarray(test_accs)))
    return train_accs, test_accs


if __name__ == "__main__":
    trsfr_args = vars(get_parser("Face Learner"))
    trsfr_args["rng"] = set_seed(sd=-1)
    run_transfer_learner_reps(exp_args=trsfr_args)
