"""Module implementing the simclr method"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optimizers
import tqdm
import numpy as np
import os
import csv
import datetime
import pickle
import utils
import torch.utils.data
import torch.backends.cudnn

import network as simclr_net
from dataset_toybox import ToyboxDataset
from dataset_core50 import data_core50
import parser

outputDirectory = "../output_trial/"
mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


def get_train_transform(tr):
    """Returns the train transform"""
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    if tr == 1:
        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size=224, padding=25),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        # GaussianBlur(kernel_size = 22),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    elif tr == 2:
        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size=224, padding=25),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        # GaussianBlur(kernel_size = 22),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    elif tr == 3:
        transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size=224, padding=25),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    
    elif tr == 4:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    elif tr == 5:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    else:
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    
    return transform


def learn_unsupervised(args, simclr_network, devices):
    """unsupervised part of the learning"""
    num_epochs = args['epochs1']
    transform_train = get_train_transform(args["transform"])
    
    if args["dataset"] == "core50":
        train_data = data_core50(root="../data", rng=args["rng"], train=True, nViews=2, size=224,
                                 transform=transform_train, fraction=args["frac1"], distort=args['distort'],
                                 adj=args['adj'],
                                 hyperTune=args["hypertune"], split_by_sess=args["sessionSplit"],
                                 distortArg=args["distortArg"])
    else:
        train_data = ToyboxDataset(root="../data", rng=args["rng"], train=True, n_views=2, size=224,
                                   transform=transform_train, fraction=args["frac1"], distort=args['distort'],
                                   adj=args['adj'],
                                   hypertune=args["hypertune"], distort_arg=args["distortArg"],
                                   interpolate=args['interpolate'])
    
    trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=args['batch_size'], shuffle=True,
                                                  num_workers=args['workers'])
    
    optimizer = optimizers.SGD(simclr_network.backbone.parameters(), lr=args["lr"], weight_decay=args["weight_decay"],
                               momentum=0.9)
    optimizer.add_param_group({'params': simclr_network.fc.parameters()})
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(1.25 * num_epochs),
                                                           eta_min=0.1 * args["lr"])
    show = False
    train_losses = []
    if args["resume"]:
        for ep in range(args["epochsRan"]):
            if ep > 8:
                scheduler.step()
    
    netParallel = nn.DataParallel(simclr_network, device_ids=devices)
    for ep in range(num_epochs):
        tqdmBar = tqdm.tqdm(trainDataLoader)
        b = 0
        avg_loss = 0.0
        for _, images, _ in tqdmBar:
            b += 1
            optimizer.zero_grad()
            images = torch.cat(images, dim=0)
            if show:
                unorm = utils.UnNormalize(mean=mean, std=std)
                im1 = transforms.ToPILImage()(unorm(images[0]))
                im1.show()
                im2 = transforms.ToPILImage()(unorm(images[args['batch_size']]))
                im2.show()
                show = False
            # images = images.to(device)
            features = netParallel(images)
            logits, labels = utils.info_nce_loss(features=features, temp=args["temperature"])
            loss = nn.CrossEntropyLoss()(logits, labels)
            avg_loss = (avg_loss * (b - 1) + loss.item()) / b
            loss.backward()
            optimizer.step()
            tqdmBar.set_description("Epoch: {:d}/{:d}, Loss: {:.6f}, LR: {:.8f}".format(ep + 1, num_epochs, avg_loss,
                                                                                        optimizer.param_groups[0][
                                                                                            'lr']))
        
        train_losses.append(avg_loss)
        if args["resume"] or ep > 8:
            scheduler.step()
        if args["saveRate"] != -1 and (ep + 1) % args["saveRate"] == 0 and args["save"]:
            fileName = args["saveName"] + "unsupervised_" + str(ep + 1) + ".pt"
            torch.save(simclr_network.state_dict(), fileName, _use_new_zipfile_serialization=False)
    if args["save"]:
        fileName = args["saveName"] + "unsupervised_final.pt"
        print("Saving network weights to", fileName)
        torch.save(simclr_network.state_dict(), fileName, _use_new_zipfile_serialization=False)
        fileName = args["saveName"] + "train_losses.pickle"
        f = open(fileName, "wb")
        pickle.dump(train_losses, f, protocol=pickle.DEFAULT_PROTOCOL)
        f.close()


def learn_supervised(args, simclr_network, devices, k):
    """Perform unsupervised learning"""
    transform_train = get_train_transform(args["transform"])
    
    transform_test = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    if args["dataset"] == "core50":
        trainSet = data_core50(root="../data", train=True, transform=transform_train, split="super", size=224,
                               fraction=args["frac2"], hyperTune=args["hypertune"], rng=args["rng"],
                               split_by_sess=args["sessionSplit"])
    else:
        if not args['transfer']:
            trainSet = ToyboxDataset(root="../data", train=True, transform=transform_train, split="super", size=224,
                                     fraction=args["frac2"], hypertune=args["hypertune"], rng=args["rng"],
                                     interpolate=args['interpolate'])
        else:
            trainSet = data_core50(root="../data", train=True, transform=transform_train, split="super",
                                   size=224, fraction=args["frac2"], hyperTune=args["hypertune"], rng=args["rng"],
                                   split_by_sess=args["sessionSplit"])
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args['batch_size'], shuffle=True,
                                              num_workers=args['workers'])
    
    if args["dataset"] == "core50":
        testSet = data_core50(root="../data", train=False, transform=transform_test, split="super", size=224,
                              hyperTune=args["hypertune"], rng=args["rng"], split_by_sess=args["sessionSplit"])
    else:
        testSet = ToyboxDataset(root="../data", train=False, transform=transform_test, split="super", size=224,
                                hypertune=args["hypertune"], rng=args["rng"])
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=args['batch_size'], shuffle=False,
                                             num_workers=args['workers'])
    if args["freeze_backbone"]:
        simclr_network.freeze_feat()
    else:
        simclr_network.freeze_head()
    pytorch_total_params = sum(p.numel() for p in simclr_network.parameters())
    pytorch_total_params_train = sum(p.numel() for p in simclr_network.parameters() if p.requires_grad)
    print(str(pytorch_total_params_train) + "/" + str(pytorch_total_params) + " parameters are trainable.")
    net = nn.DataParallel(simclr_network, device_ids=devices)
    
    optimizer = torch.optim.SGD(simclr_network.classifier_fc.parameters(), lr=args["lr_ft"],
                                weight_decay=args["weight_decay"])
    if not args["freeze_backbone"]:
        optimizer.add_param_group({'params': simclr_network.backbone.parameters()})
    
    numEpochsS = args['epochs2']
    repEval = 10
    
    for ep in range(numEpochsS):
        ep_id = 0
        tot_loss = 0
        tqdmBar = tqdm.tqdm(trainLoader)
        for _, images, labels in tqdmBar:
            optimizer.zero_grad()
            logits = net(images)
            loss = nn.CrossEntropyLoss()(logits, labels.cuda())
            tot_loss += loss.item()
            ep_id += 1
            tqdmBar.set_description("Repetition: {:d}/{:d} Epoch: {:d}/{:d} Loss: {:.4f}, LR: {:.8f}"
                                    .format(k, args["supervisedRep"], ep + 1, numEpochsS, tot_loss / ep_id,
                                            optimizer.param_groups[0]['lr']))
            loss.backward()
            optimizer.step()
        if ep % 5 == 0 and ep > 0:
            optimizer.param_groups[0]['lr'] *= 0.7
        
        if ep % repEval == repEval - 1:
            top1acc = 0
            top5acc = 0
            totTrainPoints = 0
            for _, (indices, images, labels) in enumerate(trainLoader):
                with torch.no_grad():
                    logits = net(images)
                top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1, 5))
                top1acc += top[0].item() * pred.shape[0]
                top5acc += top[1].item() * pred.shape[0]
                totTrainPoints += pred.shape[0]
            top1acc /= totTrainPoints
            top5acc /= totTrainPoints
            
            print("Train Accuracies 1 and 5:", top1acc, top5acc)
            
            top1acc = 0
            top5acc = 0
            totTestPoints = 0
            for _, (indices, images, labels) in enumerate(testLoader):
                with torch.no_grad():
                    logits = net(images)
                top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1, 5))
                top1acc += top[0].item() * indices.size()[0]
                top5acc += top[1].item() * indices.size()[0]
                totTestPoints += indices.size()[0]
            top1acc /= totTestPoints
            top5acc /= totTestPoints
            
            print("Test Accuracies 1 and 5:", top1acc, top5acc)
    
    net.eval()
    
    if args["save"]:
        fileName = args["saveName"] + "test_predictions.csv"
        csvFileTest = open(fileName, "w")
        csvWriterTest = csv.writer(csvFileTest)
        csvWriterTest.writerow(["Index", "True Label", "Predicted Label"])
        
        fileName = args["saveName"] + "train_predictions.csv"
        csvFileTrain = open(fileName, "w")
        csvWriterTrain = csv.writer(csvFileTrain)
        csvWriterTrain.writerow(["Index", "True Label", "Predicted Label"])
        
        fileName = args["saveName"] + "train_indices.pickle"
        selectedIndicesFile = open(fileName, "wb")
        pickle.dump(trainSet.indicesSelected, selectedIndicesFile, pickle.DEFAULT_PROTOCOL)
        selectedIndicesFile.close()
    
    top1acc = 0
    top5acc = 0
    totTrainPoints = 0
    for _, (indices, images, labels) in enumerate(trainLoader):
        with torch.no_grad():
            logits = net(images)
        top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1, 5))
        top1acc += top[0].item() * pred.shape[0]
        top5acc += top[1].item() * pred.shape[0]
        totTrainPoints += pred.shape[0]
        if args["save"]:
            pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
            for idx in range(pred.shape[0]):
                row = [indices[idx], labels[idx], pred[idx]]
                csvWriterTrain.writerow(row)
    top1acc /= totTrainPoints
    top5acc /= totTrainPoints
    
    print("Train Accuracies 1 and 5:", top1acc, top5acc)
    
    top1acc = 0
    top5acc = 0
    totTestPoints = 0
    for _, (indices, images, labels) in enumerate(testLoader):
        with torch.no_grad():
            logits = net(images)
        top, pred = utils.calc_accuracy(logits, labels.cuda(), topk=(1, 5))
        top1acc += top[0].item() * indices.size()[0]
        top5acc += top[1].item() * indices.size()[0]
        totTestPoints += indices.size()[0]
        if args["save"]:
            pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
            for idx in range(pred.shape[0]):
                row = [indices[idx], labels[idx], pred[idx]]
                csvWriterTest.writerow(row)
    top1acc /= totTestPoints
    top5acc /= totTestPoints
    
    print("Test Accuracies 1 and 5:", top1acc, top5acc)
    
    if args["save"]:
        fileName = args["saveName"] + "supervised.pt"
        print("Saving network weights to:", fileName)
        torch.save(simclr_network.state_dict(), fileName, _use_new_zipfile_serialization=False)
        csvFileTrain.close()
        csvFileTest.close()
    
    return top1acc, top5acc


def set_seed(sd):
    """Set the seed for the experiments
    """
    if sd == -1:
        sd = np.random.randint(0, 65536)
    print("Setting seed to", sd)
    torch.manual_seed(sd)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(sd)
    return rng


def train_unsupervised_and_supervised(args):
    """Train the networks"""
    print(torch.cuda.get_device_name(0))
    numGPUs = torch.cuda.device_count()
    deviceIDs = [i for i in range(numGPUs)]
    args["start"] = datetime.datetime.now()
    rng = set_seed(args["seed"])
    args["rng"] = rng
    if args["saveName"] == "":
        if args["distort"] == "transform":
            args["saveName"] = "trained_model_cropped_" + args["distort"] + "_" + str(args["adj"]) + "/"
        else:
            args["saveName"] = "trained_model_cropped_" + args["distort"] + "/"
    args["saveName"] = outputDirectory + args["saveName"]
    if args['save']:
        os.makedirs(args['saveName'], exist_ok=False)
    # device = torch.device('cuda:0')
    if args["dataset"] == "toybox":
        network = simclr_net.SimClRNet(num_classes=12).cuda()
    else:
        network = simclr_net.SimClRNet(num_classes=10).cuda()
    if args["resume"]:
        if args["resumeFile"] == "":
            raise RuntimeError("No file provided for model to start from.")
        if args["epochsRan"] == -1:
            raise RuntimeError("Specify number of epochs ran for model which should be trained further.")
        network.load_state_dict(torch.load(outputDirectory + args["resumeFile"]))
        args["saveName"] = outputDirectory + args["resumeFile"] + "/"
    network.freeze_classifier()
    if args["save"]:
        configFileName = args["saveName"] + "config.pickle"
        configFile = open(configFileName, "wb")
        pickle.dump(args, configFile, pickle.DEFAULT_PROTOCOL)
        # print(args, file = configFile)
        configFile.close()
    
    if args['epochs1'] > 0:
        pytorch_total_params = sum(p.numel() for p in network.parameters())
        pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(str(pytorch_total_params_train) + "/" + str(pytorch_total_params) + " parameters are trainable.")
        learn_unsupervised(args=args, simclr_network=network, devices=deviceIDs)
    
    pytorch_total_params = sum(p.numel() for p in network.parameters())
    pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(str(pytorch_total_params_train) + "/" + str(pytorch_total_params) + " parameters are trainable.")
    network.unsupervised = False
    learn_supervised(args=args, simclr_network=network, devices=deviceIDs, k=1)
    if args['save']:
        calculate_train_test_activations(args=args, simclr_network=network)
    
    
def get_activations(network, loader):
    """Run data through network and get activations"""
    targets = []
    preds = []
    network.freeze_all_params()
    # switch to evaluate mode
    network.eval()

    with torch.no_grad():
        for i, (_, images, target) in enumerate(loader):
            images = images.cuda()
        
            # compute predictions
            pred = network.forward_l4(images)
            pred = torch.mean(pred, dim=(2, 3))
        
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())

    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)
    return targets, preds
    
    
def calculate_train_test_activations(args, simclr_network):
    """Computes the neuron selectivities"""
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])
    train_set = ToyboxDataset(root="../data", train=True, transform=transform_train, split="super", size=224,
                              fraction=0.5, hypertune=True, rng=np.random.default_rng(0),
                              interpolate=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)
    targets, preds = get_activations(network=simclr_network, loader=train_loader)
    np.save(args['saveName'] + "train_activations.npy", preds)
    np.save(args['saveName'] + "train_targets.npy", targets)

    test_set = ToyboxDataset(root="../data", train=False, transform=transform_train, split="super", size=224,
                             fraction=0.5, hypertune=True, rng=np.random.default_rng(0),
                             interpolate=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    targets, preds = get_activations(network=simclr_network, loader=test_loader)
    np.save(args['saveName'] + "test_activations.npy", preds)
    np.save(args['saveName'] + "test_targets.npy", targets)
    

def evaluate_trained_network(args):
    """Evaluate the trained network"""
    num_gpus = torch.cuda.device_count()
    device_ids = [i for i in range(num_gpus)]
    args["start"] = datetime.datetime.now()
    args["saveName"] = outputDirectory + args["saveName"]
    if not args["resume"]:
        raise RuntimeError("Set resume flag and specify running file")
    if args["supervisedRep"] < 1:
        raise RuntimeError("Number of repetitions for supervised training must be > 0. Use \'-rep\' option to set.")
    if args["resume"]:
        if args["resumeFile"] == "":
            raise RuntimeError("No file provided for model to start from.")
        if args["saveName"] == "":
            args["saveName"] = outputDirectory + args["resumeFile"]
    save_name = args["saveName"]
    accuracies = []
    args["seed"] = -1
    for i in range(args["supervisedRep"]):
        print("------------------------------------------------------------------------------------------------")
        print("Repetition " + str(i + 1) + " of " + str(args["supervisedRep"]))
        args["saveName"] = save_name + "_" + str(i + 1) + "_"
        rng = set_seed(args["seed"])
        args["rng"] = rng
        if args["dataset"] == "toybox":
            network = simclr_net.SimClRNet(num_classes=12).cuda()
        else:
            network = simclr_net.SimClRNet(num_classes=10).cuda()
        network.load_state_dict(torch.load(outputDirectory + args["resumeFile"]))
        print("Loaded network weights from", args["resumeFile"])
        network.freeze_classifier()
        network.unsupervised = False
        if args["save"]:
            config_file_name = args["saveName"] + "config.pickle"
            configFile = open(config_file_name, "wb")
            pickle.dump(args, configFile, pickle.DEFAULT_PROTOCOL)
            # print(args, file = configFile)
            configFile.close()
        
        pytorch_total_params = sum(p.numel() for p in network.parameters())
        pytorch_total_params_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(str(pytorch_total_params_train) + "/" + str(pytorch_total_params) + " parameters are trainable.")
        top1, top5 = learn_supervised(args=args, simclr_network=network, devices=device_ids, k=i + 1)
        accuracies.append(top1)
        print("------------------------------------------------------------------------------------------------")
    print("The accuracies on the test set are:", accuracies)
    print("Mean accuracy on test set is", np.mean(np.asarray(accuracies)))
    print("Std. deviation of accuracy on test set is", np.std(np.asarray(accuracies)))
    if args["save"]:
        fileName = save_name + "_test_accuracies.csv"
        acc_file = open(fileName, "w")
        csv_acc = csv.writer(acc_file)
        for acc in accuracies:
            csv_acc.writerow([acc])
        acc_file.close()
    return np.mean(np.asarray(accuracies))


if __name__ == "__main__":
    if not os.path.isdir(outputDirectory):
        os.mkdir(outputDirectory)
    simclr_args = vars(parser.get_parser("SimCLR Parser"))
    assert (simclr_args["dataset"] == "core50" or simclr_args["dataset"] == "toybox")
    simclr_args['saveName'] += '/'
    saveName = simclr_args["saveName"]
    num_reps = simclr_args["supervisedRep"]
    simclr_args["supervisedRep"] = 1
    train_unsupervised_and_supervised(args=simclr_args)
    if num_reps > 0:
        simclr_args["resume"] = True
        simclr_args["resumeFile"] = saveName + "unsupervised_final.pt"
        simclr_args["saveName"] = saveName + "eval"
        simclr_args["supervisedRep"] = num_reps
        evaluate_trained_network(args=simclr_args)
