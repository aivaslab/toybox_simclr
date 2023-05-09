"""Module implementing the network for SimCLR"""
import torchvision.models as models
import torch.nn as nn
import copy
import numpy as np


def init_weights(m):
	"""Weight init for modules"""
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)


class Identity(nn.Module):
	"""Module implementing the Identity layer"""
	
	def __init__(self):
		super(Identity, self).__init__()
	
	def __call__(self, x):
		return x


class SimClRNet(nn.Module):
	"""Module implementing network architecture for simclr"""
	
	def __init__(self, num_classes):
		super().__init__()
		self.backbone = models.resnet18(pretrained=False, num_classes=256)
		feat_num = self.backbone.fc.in_features
		self.fc = nn.Sequential(nn.Linear(feat_num, feat_num), nn.ReLU(inplace=True), nn.Linear(feat_num, 128))
		self.backbone.fc = Identity()
		self.feat_num = feat_num
		self.classifier_fc = nn.Linear(self.feat_num, num_classes)
		self.unsupervised = True
	
	def forward(self, x):
		"""Forward method"""
		y = self.backbone(x)
		if self.unsupervised:
			y = self.fc(y)
		else:
			y = self.classifier_fc(y)
		return y
	
	def forward_l1(self, x):
		"""Forward_l1"""
		y = self.backbone.conv1(x)
		y = self.backbone.layer1(y)
		return y
	
	def forward_l2(self, x):
		"""Forward_l2"""
		y = self.backbone.conv1(x)
		y = self.backbone.layer1(y)
		y = self.backbone.layer2(y)
		return y
	
	def forward_l3(self, x):
		"""Forward_l3"""
		y = self.backbone.conv1(x)
		y = self.backbone.layer1(y)
		y = self.backbone.layer2(y)
		y = self.backbone.layer3(y)
		return y
	
	def forward_l4(self, x):
		"""Forward_l4"""
		y = self.backbone.conv1(x)
		y = self.backbone.layer1(y)
		y = self.backbone.layer2(y)
		y = self.backbone.layer3(y)
		y = self.backbone.layer4(y)
		return y
	
	def freeze_feat(self):
		"""Freeze the feature extractors"""
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		self.backbone.eval()
		self.fc.eval()
		self.classifier_fc.train()
		print("Freezing backbone and unsupervised head.")
	
	def freeze_classifier(self):
		"""Freeze the classifier"""
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = True
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		self.backbone.train()
		self.fc.train()
		self.classifier_fc.eval()
		print("Freezing classifier fc.")
	
	def freeze_head(self):
		"""Freeze the nonlinear head"""
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		self.backbone.train()
		self.fc.eval()
		self.classifier_fc.train()
		print("Freezing only unsupervised head fc.")
	
	def freeze_backbone_layer_1(self):
		"""Freeze the layer 1 of backbone"""
		for name, param in self.backbone.conv1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.bn1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.relu.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.maxpool.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer2.named_parameters():
			param.requires_grad = True
		for name, param in self.backbone.layer3.named_parameters():
			param.requires_grad = True
		for name, param in self.backbone.layer4.named_parameters():
			param.requires_grad = True
		for name, param in self.backbone.avgpool.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		self.classifier_fc.eval()
		self.fc.eval()
		self.backbone.conv1.eval()
		self.backbone.bn1.eval()
		self.backbone.relu.eval()
		self.backbone.maxpool.eval()
		self.backbone.layer1.eval()
		self.backbone.layer2.train(mode=True)
		self.backbone.layer3.train(mode=True)
		self.backbone.layer4.train(mode=True)
		print(
			"Freezing initial conv layer of ResNet and first residual block. Both heads are frozen and remaining "
			"residual blocks are unfrozen."
		)
	
	def freeze_backbone_layer_2(self):
		"""Freeze backbone upto layer2"""
		for name, param in self.backbone.conv1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.bn1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.relu.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.maxpool.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer2.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer3.named_parameters():
			param.requires_grad = True
		for name, param in self.backbone.layer4.named_parameters():
			param.requires_grad = True
		for name, param in self.backbone.avgpool.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		self.classifier_fc.eval()
		self.fc.eval()
		self.backbone.conv1.eval()
		self.backbone.bn1.eval()
		self.backbone.relu.eval()
		self.backbone.maxpool.eval()
		self.backbone.layer1.eval()
		self.backbone.layer2.eval()
		self.backbone.layer3.train(mode=True)
		self.backbone.layer4.train(mode=True)
		print(
			"Freezing initial conv layer of ResNet and first two residual blocks. Both heads are frozen and remaining "
			"residual blocks are unfrozen.")
	
	def freeze_backbone_layer_3(self):
		"""Freeze backbone upto layer3"""
		for name, param in self.backbone.conv1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.bn1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.relu.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.maxpool.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer1.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer2.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer3.named_parameters():
			param.requires_grad = False
		for name, param in self.backbone.layer4.named_parameters():
			param.requires_grad = True
		for name, param in self.backbone.avgpool.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		self.classifier_fc.eval()
		self.fc.eval()
		self.backbone.conv1.eval()
		self.backbone.bn1.eval()
		self.backbone.relu.eval()
		self.backbone.maxpool.eval()
		self.backbone.layer1.eval()
		self.backbone.layer2.eval()
		self.backbone.layer3.eval()
		self.backbone.layer4.train(mode=True)
		print(
			"Freezing initial conv layer of ResNet and first three residual blocks. Both heads are frozen and remaining "
			"residual blocks are unfrozen.")
	
	def unfreeze_all_params(self):
		"""Unfreeze all parameters"""
		for name, param in self.backbone.named_parameters():
			param.requires_grad = True
		for name, param in self.fc.named_parameters():
			param.requires_grad = True
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		self.backbone.train()
		self.fc.train()
		self.classifier_fc.train()
		print("Unfreezing all params in SimCLR network. All params should be trainable.")
	
	def freeze_all_params(self):
		"""Freeze all parameters"""
		for name, param in self.backbone.named_parameters():
			param.requires_grad = False
		for name, param in self.fc.named_parameters():
			param.requires_grad = False
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		self.backbone.eval()
		self.fc.eval()
		self.classifier_fc.eval()
		print("Freezing all params in SimCLR network. No params should be trainable.")


class BYOLNet(nn.Module):
	
	def __init__(self, numClasses, beta=0.996):
		super().__init__()
		self.numClasses = numClasses
		self.encoder_backbone = models.resnet18(pretrained=False, num_classes=256)
		self.num_features = self.encoder_backbone.fc.in_features
		self.start_beta = beta
		self.beta = self.start_beta
		feat_num = self.num_features
		self.encoder_backbone.fc = nn.Identity()
		self.encoder_projection = nn.Sequential(nn.Linear(in_features=feat_num, out_features=2 * feat_num),
												nn.BatchNorm1d(num_features=2 * feat_num), nn.ReLU(inplace=True),
												nn.Linear(in_features=2 * feat_num, out_features=128))
		self.encoder_prediction = nn.Sequential(nn.Linear(in_features=128, out_features=2 * feat_num),
												nn.BatchNorm1d(num_features=2 * feat_num), nn.ReLU(inplace=True),
												nn.Linear(in_features=2 * feat_num, out_features=128))
		
		self.classifier_fc = nn.Linear(feat_num, numClasses)
		self.encoder_backbone.apply(init_weights)
		self.encoder_projection.apply(init_weights)
		self.encoder_prediction.apply(init_weights)
		self.classifier_fc.apply(init_weights)
		self.target_backbone = copy.deepcopy(self.encoder_backbone)
		self.target_projection = copy.deepcopy(self.encoder_projection)
	
	def update_momentum(self, epochs, total_epochs):
		self.beta = 1 - (1 - self.start_beta) * (np.cos(np.pi * epochs / total_epochs) + 1) / 2.0
	
	def update_target_network(self):
		for current_params, ma_params in zip(self.encoder_backbone.parameters(), self.target_backbone.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = old_weight * self.beta + up_weight * (1 - self.beta)
		
		for current_params, ma_params in zip(self.encoder_projection.parameters(), self.target_projection.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = old_weight * self.beta + up_weight * (1 - self.beta)
	
	def encoder_forward(self, x):
		y = self.encoder_backbone(x)
		# print(y.shape)
		y = self.encoder_projection(y)
		y = self.encoder_prediction(y)
		
		return y
	
	def target_forward(self, x):
		y = self.target_backbone(x)
		y = self.target_projection(y)
		
		return y
	
	def classify(self, x):
		y = self.encoder_backbone(x)
		y = self.classifier_fc(y)
		
		return y
	
	def freeze_encoder_backbone(self):
		for name, param in self.encoder_backbone.named_parameters():
			param.requires_grad = False
		self.encoder_backbone.eval()
		print("Freezing encoder network backbone.....")
	
	def unfreeze_encoder_backbone(self):
		for name, param in self.encoder_backbone.named_parameters():
			param.requires_grad = True
		self.encoder_backbone.train()
		print("Unfreezing encoder network backbone.....")
	
	def freeze_encoder_projection(self):
		for name, param in self.encoder_projection.named_parameters():
			param.requires_grad = False
		self.encoder_projection.eval()
		print("Freezing encoder network projection.....")
	
	def unfreeze_encoder_projection(self):
		for name, param in self.encoder_projection.named_parameters():
			param.requires_grad = True
		self.encoder_projection.train()
		print("Unfreezing encoder network projection.....")
	
	def freeze_encoder_prediction(self):
		for name, param in self.encoder_prediction.named_parameters():
			param.requires_grad = False
		self.encoder_prediction.eval()
		print("Freezing encoder network prediction.....")
	
	def unfreeze_encoder_prediction(self):
		for name, param in self.encoder_prediction.named_parameters():
			param.requires_grad = True
		self.encoder_prediction.train()
		print("Unfreezing encoder network prediction.....")
	
	def freeze_target_backbone(self):
		for name, param in self.target_backbone.named_parameters():
			param.requires_grad = False
		self.target_backbone.eval()
		print("Freezing target network backbone.....")
	
	def unfreeze_target_backbone(self):
		for name, param in self.target_backbone.named_parameters():
			param.requires_grad = True
		self.target_backbone.train()
		print("Unfreezing target network backbone.....")
	
	def freeze_target_projection(self):
		for name, param in self.target_projection.named_parameters():
			param.requires_grad = False
		self.target_projection.eval()
		print("Freezing target network projection.....")
	
	def unfreeze_target_projection(self):
		for name, param in self.target_projection.named_parameters():
			param.requires_grad = True
		self.target_projection.train()
		print("Unfreezing target network projection.....")
	
	def freeze_classifier(self):
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = False
		self.classifier_fc.eval()
		print("Freezing classifier fc.....")
	
	def unfreeze_classifier(self):
		for name, param in self.classifier_fc.named_parameters():
			param.requires_grad = True
		self.classifier_fc.train()
		print("Unfreezing classififer fc.....")
