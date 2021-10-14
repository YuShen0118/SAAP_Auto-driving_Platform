import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import math
import glob
import torchvision.models as models
from sys import platform
import random

import os
ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)
DATASET_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB/"

class MyDataset(torch.utils.data.Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, train_list_base, n_class=4, transform=None, is_hint=True):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.train_list_base = train_list_base
		self.transform = transform
		self.n_class = n_class
		self.is_hint = is_hint

	def __len__(self):
		return len(self.train_list_base)

	def __getitem__(self, idx):
		img_path = self.train_list_base[idx]

		label = int(os.path.basename(img_path).replace(".jpg","")) % self.n_class

		# img_path = img_path.replace("trainB", "trainB_blur_"+str(label+1))
		# img_path = img_path.replace("valB", "valB_blur_"+str(label+1))
		# img_path = img_path.replace("valHc", "valHc_blur_"+str(label+1))
		# img_path = img_path.replace("valAds", "valAds_blur_"+str(label+1))
		# img_path = img_path.replace("valB", "valB_combined_5_0")
		# img_path = img_path.replace("valB", "valB_combined_5levels_blurfirst_"+str(label+1))
		# img_path = img_path.replace("valB", "valB_combined_5levels_blurlast_"+str(label+1))
		
		if not os.path.isfile(img_path):
			print(img_path, " not exists")

		img = Image.open(img_path)
		# img = img.convert("RGB")
		# img = img.resize((200, 66))
		# img.show()

		if self.transform:
			img = self.transform(img)

		if self.is_hint:
			img1 = Image.open(img_path.replace("major", "hint"))
			if self.transform:
				img1 = self.transform(img1)
			img = torch.cat((img, img1), 0)


		# label = np.array([label])
		label = torch.tensor(label)

		return img, label, img_path


class Net_classification(nn.Module):
	def __init__(self, nChannel=3, nClass=8):
		super(Net_classification, self).__init__()
		self.conv1 = nn.Conv2d(nChannel, 3, 1)
		self.resnet = models.resnet18()
		self.fc1 = nn.Linear(1000, 500)
		self.fc2 = nn.Linear(500, 100)
		self.fc3 = nn.Linear(100, nClass)

	def forward(self, x):
		# x = LambdaLayer(lambda x: x/127.5 - 1.0)(x)
		x = self.conv1(x)
		feature = self.resnet(x)
		y = F.relu(self.fc1(feature))
		y = F.relu(self.fc2(y))
		output = self.fc3(y)
		return output, feature


class Net_feature(nn.Module):
	def __init__(self, nChannel=3):
		super(Net_feature, self).__init__()
		self.conv1 = nn.Conv2d(nChannel, 3, 1)
		self.resnet = models.resnet18()

	def forward(self, x):
		# x = LambdaLayer(lambda x: x/127.5 - 1.0)(x)
		x = self.conv1(x)
		feature = self.resnet(x)
		return feature

class Net_classifier(nn.Module):
	def __init__(self, nClass=8):
		super(Net_classifier, self).__init__()
		self.fc1 = nn.Linear(1000, 500)
		self.fc2 = nn.Linear(500, 100)
		self.fc3 = nn.Linear(100, nClass)

	def forward(self, x):
		y = F.relu(self.fc1(x))
		y = F.relu(self.fc2(y))
		output = self.fc3(y)
		return output


def train(net, device, trainloader, valloader, out_path):
	# net = Net()
	net.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	nEpoch = 200
	for epoch in range(nEpoch):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			# inputs, labels = data
			inputs, labels, imgpaths = data[0].to(device), data[1].to(device), data[2]
			
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs,_ = net(inputs)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()

		print('[%d, %5d] loss: %.3f' %
			  (epoch + 1, nEpoch, running_loss))

		if epoch % 20 == 0:	# print every 2000 mini-batches
			torch.save(net.state_dict(), out_path)
			test(net, out_path, device, trainloader)
			test(net, out_path, device, valloader)

	print('Finished Training')

	torch.save(net.state_dict(), out_path)


def train_hint(net, device, trainloader, trainloader_hint, valloader, out_path, pre_model_path=""):
	# net = Net()
	net_f_ori, net_f_hint, net_d = net

	if pre_model_path != "":
		net_f_ori.load_state_dict(torch.load(pre_model_path.replace(".pth", "_fo.pth")))
		net_f_hint.load_state_dict(torch.load(pre_model_path.replace(".pth", "_fh.pth")))
		net_d.load_state_dict(torch.load(pre_model_path.replace(".pth", "_d.pth")))

	net_f_ori.to(device)
	net_f_hint.to(device)
	net_d.to(device)

	loss_classification = nn.CrossEntropyLoss()
	loss_regression = nn.MSELoss()

	optimizer_fo = optim.Adam(net_f_ori.parameters(), lr=0.001)
	optimizer_fh = optim.Adam(net_f_hint.parameters(), lr=0.001)
	optimizer_d = optim.Adam(net_d.parameters(), lr=0.001)

	nRound = 200
	nEpoch = 1

	acc_max = 0

	for round_id in range(nRound):
		for epoch in range(nEpoch):  # loop over the dataset multiple times
			running_loss = 0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				# inputs, labels = data
				inputs, labels = data[0].to(device), data[1].to(device)
				
				# zero the parameter gradients
				optimizer_fo.zero_grad()
				optimizer_d.zero_grad()

				# forward + backward + optimize
				# feature = net_f_ori(inputs[:,0:3,:,:])
				feature = net_f_ori(inputs)
				outputs = net_d(feature)

				loss = loss_classification(outputs, labels)
				loss.backward()
				optimizer_fo.step()
				optimizer_d.step()

				# print statistics
				running_loss += loss.item()
			print('[%d, %5d, %d, %5d] loss1: %.3f' %
				  (round_id + 1, nRound, epoch + 1, nEpoch, running_loss))

		# for param in net_f_hint.resnet.parameters():
		# 	print(param.data)
		# 	break

		net_f_hint.resnet.load_state_dict(net_f_ori.resnet.state_dict())
		net_f_hint.conv1.weight.data = torch.zeros(net_f_hint.conv1.weight.data.shape)
		net_f_hint.conv1.weight.data[:,0:3,:,:] = net_f_ori.conv1.weight.data
		net_f_hint.conv1.bias.data = net_f_ori.conv1.bias.data
		net_f_hint.to(device)

		for epoch in range(nEpoch):  # loop over the dataset multiple times
			running_loss_1 = 0.0
			running_loss_2 = 0.0
			running_loss_3 = 0.0
			for i, data in enumerate(trainloader_hint, 0):
				# get the inputs; data is a list of [inputs, labels]
				# inputs, labels = data
				inputs, labels = data[0].to(device), data[1].to(device)

				# zero the parameter gradients
				optimizer_fo.zero_grad()
				optimizer_fh.zero_grad()
				optimizer_d.zero_grad()
				
				input_base = inputs[:,0:3,:,:]
				input_hint = inputs

				feature_base = net_f_ori(input_base)
				output_base = net_d(feature_base)

				feature_hint = net_f_hint(input_hint)
				output_hint = net_d(feature_hint)


				# ************************ teacher student learn together ************************
				# loss_1 = loss_classification(output_base, labels)
				# loss_2 = loss_classification(output_hint, labels)
				# loss_3 = loss_regression(feature_base, feature_hint)

				# loss = loss_1 + loss_2 + loss_3
				# loss.backward()

				# optimizer_fo.step()
				# optimizer_fh.step()
				# optimizer_d.step()


				# ************************ teacher first, then student ************************
				# teacher learn first
				loss_2 = loss_classification(output_hint, labels)
				loss_2.backward(retain_graph=True)
				optimizer_fh.step()

				# student learn next
				loss_1 = loss_classification(output_base, labels)
				loss_3 = loss_regression(feature_base, feature_hint)

				loss = loss_1 + loss_3
				loss.backward()

				optimizer_fo.step()
				optimizer_d.step()

				# print statistics
				running_loss_1 += loss_1.item()
				running_loss_2 += loss_2.item()
				running_loss_3 += loss_3.item()

			print('[%d, %5d, %d, %5d] loss1: %.3f loss2: %.3f loss3: %.3f' %
				  (round_id + 1, nRound, epoch + 1, nEpoch, running_loss_1, running_loss_2, running_loss_3))

		if round_id % 20 == 0 or round_id == nRound-1:	# print every 2000 mini-batches
			torch.save(net_f_ori.state_dict(), out_path.replace(".pth", "_fo.pth"))
			torch.save(net_f_hint.state_dict(), out_path.replace(".pth", "_fh.pth"))
			torch.save(net_d.state_dict(), out_path.replace(".pth", "_d.pth"))
			# test([net_f_ori, net_d], out_path, device, trainloader, is_hint=True)
			acc = test([net_f_ori, net_d], out_path, device, valloader, is_hint=True)
			if acc_max < acc:
				acc_max = acc

	print('Finished Training')
	return acc_max


def test(net, model_path, device, testloader, n_class=4, is_hint=False):
	counts_table=np.zeros((n_class, n_class))

	class_correct = list(0. for i in range(n_class))
	class_total = list(0. for i in range(n_class))
	with torch.no_grad():
		features_all = []
		labels_all = []
		for data in testloader:
			# images, labels = data
			images, labels = data[0].to(device), data[1].to(device)
			# print(labels)
			if not is_hint:
				outputs,feature = net(images)
			else:
				net_f_ori, net_d = net
				images = images[:,0:3,:,:]
				feature = net_f_ori(images)
				outputs = net_d(feature)

			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()

			for i in range(labels.shape[0]):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1
				counts_table[label][predicted[i]] += 1

			features_all.append(feature.cpu().detach())
			labels_all.append(labels.cpu().detach())

	for i in range(n_class):
		if class_total[i] > 0:
			print('Accuracy of level %5s : %d/%d = %2d %%' % (
				i, class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))

	print(counts_table)

	print('Overall Accuracy: %.2f %%' % (np.sum(class_correct)/np.sum(class_total)*100))

	# save features
	features_all = np.concatenate(features_all)
	labels_all = np.concatenate(labels_all)
	with open('features.npy', 'wb') as f:
		np.save(f, features_all)
	with open('labels.npy', 'wb') as f:
		np.save(f, labels_all)
	# for i in range(n_class):
	#	 mask = labels_all == i
	#	 feature_i = features_all[mask]
	#	 print(feature_i.shape)
	#	 # with open(model_path.replace(".pth", "")+"_class_"+str(i)+'.npy', 'wb') as f:
	#	 with open(str(i)+'.npy', 'wb') as f:
	#		 np.save(f, feature_i)

	return np.sum(class_correct)/np.sum(class_total)*100



def add_circle(img, color, circle_list, r_min=5, r_max=10, rd_cnt_max=200, obj_type=0):
	h = img.shape[0]
	w = img.shape[1]
	for rpt in range(rd_cnt_max):
		x_new = int(np.random.rand() * (w-1))
		y_new = int(np.random.rand() * (h-1))
		r_new_max = min(r_max, x_new, y_new, w-1-x_new, h-1-y_new)
		if r_new_max < r_min:
			continue

		for circle in circle_list:
			x,y,r = circle
			r_1 = math.sqrt((x-x_new)*(x-x_new) + (y-y_new)*(y-y_new))-r
			if r_new_max > r_1:
				r_new_max = r_1

			if r_new_max < r_min:
				break

		if r_new_max < r_min:
			continue

		r_new = int(np.random.rand()*(r_new_max-r_min) + r_min)

		circle_list.append([x_new, y_new, r_new])

		if obj_type == 0:
			cv2.circle(img, (x_new, y_new), radius=r_new, color=color, thickness=-1)
		else:
			r = r_new / 1.414
			cv2.rectangle(img, (int(x_new-r), int(y_new-r)), (int(x_new+r), int(y_new+r)), color=color, thickness=-1)


		break



def generate_image_with_random_circles(n_target, n_max=20, w=64, h=64, r_min=3, r_max=10):
	img = (np.ones((h,w,3))*255).astype(np.uint8)
	circle_list = []

	color_list = [(255,0,0), (255,0,255), (255,255,0), (0,255,255), (0,255,0), (0,0,255)]

	for i in range(n_target):
		# target circle
		color = color_list[-1]
		add_circle(img, color, circle_list)
		# print(circle_list)

	img_hint = img.copy()

	for i in range(n_max-n_target):
		color_id = int(np.random.rand()*(len(color_list)-1))
		color = color_list[color_id]
		add_circle(img, color, circle_list)
		# print(circle_list)

	return img,img_hint

	# cv2.imshow("img", img)
	# cv2.waitKey(0)


def generate_image_with_random_objects(n_target, n_max=20, w=64, h=64, r_min=2, r_max=6):
	img = (np.ones((h,w,3))*255).astype(np.uint8)
	circle_list = []

	color_list = [(255,0,0), (255,0,255), (255,255,0), (0,255,255), (0,255,0), (0,0,255)]

	for i in range(n_target):
		# target: red circle
		color = color_list[-1]
		add_circle(img, color, circle_list, obj_type=0)
		# print(circle_list)

	img_hint = img.copy()

	for i in range(n_max-n_target):
		if random.randrange(2) == 0:
			color_id = int(np.random.rand()*(len(color_list)-1))
			color = color_list[color_id]
			add_circle(img, color, circle_list, obj_type=0)
		else:
			color_id = int(np.random.rand()*(len(color_list)))
			color = color_list[color_id]
			add_circle(img, color, circle_list, obj_type=1)
		# print(circle_list)

	return img,img_hint

	# cv2.imshow("img", img)
	# cv2.waitKey(0)

def generate_datasets(folder, n=10000, n_class=4):
	for i in range(n):
		print(i)
		# img,img_hint = generate_image_with_random_circles(i%n_class)
		img,img_hint = generate_image_with_random_objects(i%n_class)
		cv2.imwrite(folder+"/major/"+str(i)+".jpg", img)
		cv2.imwrite(folder+"/hint/"+str(i)+".jpg", img_hint)



def main():

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# Assuming that we are on a CUDA machine, this should print a CUDA device:
	print(device)

	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	batch_size = 128

	train_folder = "dataset1/major/"
	xList = glob.glob(train_folder+"*.jpg")

	xList = sorted (xList, key = lambda x: (len (x), x))

	xList_train, xList_val = xList[0:400], xList[5000:7000]

	# xList_train_hint = xList[3:400:4]
	# xList_train = np.concatenate((xList[0:400:4], xList[1:400:4], xList[2:400:4], xList_train_hint))
	xList_train = sorted (xList_train, key = lambda x: (len (x), x))

	# xList_train_hint = xList_train[0:80]
	xList_train_hint = xList_train


	num_workers = 8
	if platform == "win32":
		num_workers = 0

	is_hint = True
	n_class = 4

	trainset = MyDataset(train_list_base=xList_train, n_class=n_class, transform=transform, is_hint=False)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=num_workers)

	trainset_hint = MyDataset(train_list_base=xList_train_hint, n_class=n_class, transform=transform, is_hint=True)
	trainloader_hint = torch.utils.data.DataLoader(trainset_hint, batch_size=batch_size,
											  shuffle=True, num_workers=num_workers)

	testset = MyDataset(train_list_base=xList_val, n_class=n_class, transform=transform, is_hint=False)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											 shuffle=False, num_workers=num_workers)

	if is_hint:
		# hintnet
		pre_model_path = 'checkpoints/toy_hint_pretrain_base.pth'
		model_path = 'checkpoints/toy_hint_retrain.pth'
		# pre_model_path = ""
		# model_path = 'checkpoints/toy_hint_pretrain_base.pth'
		acc_list = []
		for i in range(20):
			net_f_ori = Net_feature(3)
			net_f_hint = Net_feature(6)
			net_d = Net_classifier(n_class)
			acc = train_hint([net_f_ori, net_f_hint, net_d], device, trainloader, trainloader_hint, testloader, model_path, pre_model_path=pre_model_path)
			acc_list.append(acc)
			print(np.array(acc_list))
			# if acc < 55:
			# 	break

		print(np.mean(np.array(acc_list)))
	else:
		# general
		model_path = 'checkpoints/toy_base_500_test.pth'

		acc_list = []
		for i in range(10):
			net = Net_classification(nChannel=6)
			train(net, device, trainloader, testloader, model_path)

			net = Net_classification(nChannel=6)
			net.load_state_dict(torch.load(model_path))
			net.to(device)
			acc = test(net, model_path, device, testloader)
			acc_list.append(acc)
			print(np.array(acc_list))

		print(np.mean(np.array(acc_list)))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='batch train test')
	parser.add_argument('--gpu_id', required=False, metavar="gpu_id", help='gpu id (0/1)')
	args = parser.parse_args()

	if (args.gpu_id != None):
		os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
		print("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])

	main()
	# generate_datasets("dataset1", n_class=4)
