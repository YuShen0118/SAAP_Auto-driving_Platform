import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def create_nvidia_network_pytorch(BN_flag, fClassifier, nClass, nChannel=3, Maxup_flag=False):
	#default
	if BN_flag == 0:
		return net_nvidia_pytorch()
	elif BN_flag == 3:
		return net_nvidia_featshift_pytorch()
		
	return net_nvidia_pytorch()


class net_nvidia_pytorch(nn.Module):
	def __init__(self):
		super(net_nvidia_pytorch, self).__init__()
		self.conv1 = nn.Conv2d(3, 24, 5, 2)
		self.conv2 = nn.Conv2d(24, 36, 5, 2)
		self.conv3 = nn.Conv2d(36, 48, 5, 2)
		self.conv4 = nn.Conv2d(48, 64, 3)
		self.conv5 = nn.Conv2d(64, 64, 3)
		self.fc1 = nn.Linear(64 * 1 * 18, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 10)
		self.fc4 = nn.Linear(10, 1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		#print(x.shape)
		x = x.view(-1, 64 * 1 * 18)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x


class net_nvidia_featshift_pytorch(nn.Module):
	def __init__(self):
		super(net_nvidia_featshift_pytorch, self).__init__()
		self.conv1 = nn.Conv2d(3, 24, 5, 2)
		self.conv2 = nn.Conv2d(24, 36, 5, 2)
		self.conv3 = nn.Conv2d(36, 48, 5, 2)
		self.conv4 = nn.Conv2d(48, 64, 3)
		self.conv5 = nn.Conv2d(64, 64, 3)
		self.fc1 = nn.Linear(64 * 1 * 18, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 10)
		self.fc4 = nn.Linear(10, 1)

	def forward(self, img, feature, mean2, std2):
		#img, feature = inputs
		# print(img.shape)
		# print(feature.shape)

		#print('-------------------------------------------------------')
		# print(img.cpu().detach().numpy())
		# print(feature.cpu().detach().numpy())
		# print(mean2.cpu().detach().numpy())
		# print(std2.cpu().detach().numpy())


		x = F.relu(self.conv1(img))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.conv5(x)

		#print(x.cpu().detach().numpy())

		mean1 = torch.mean(x)
		std1 = torch.std(x)

		# mean2 = torch.mean(feature)
		# std2 = torch.std(feature)

		# print(feature.shape)
		# print(mean2.shape)

		f = torch.sub(feature, mean2)
		f = torch.div(f, std2)
		f = torch.mul(f, std1)
		f = torch.add(f, mean1)

		x = F.relu(x)
		#print(x.shape)
		x = x.view(-1, 64 * 1 * 18)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)

		#print(x.cpu().detach().numpy())

		f = F.relu(f)
		#print(x.shape)
		f = f.view(-1, 64 * 1 * 18)
		f = F.relu(self.fc1(f))
		f = F.relu(self.fc2(f))
		f = F.relu(self.fc3(f))
		f = self.fc4(f)

		#print(f.cpu().detach().numpy())

		x = torch.cat((x,f),0)
		# print(out.shape)

		#print(x.shape)

		return x