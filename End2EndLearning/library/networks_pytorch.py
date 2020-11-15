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
	elif BN_flag == 4:
		return net_nvidia_pytorch_DANN()
		
	return net_nvidia_pytorch()

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

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
		x = LambdaLayer(lambda x: x/127.5 - 1.0)(x)
		x = F.elu(self.conv1(x))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
		x = F.elu(self.conv5(x))
		#print(x.shape)
		x = x.reshape(-1, 64 * 1 * 18)
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
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


		x = LambdaLayer(lambda x: x/127.5 - 1.0)(img)
		x = F.elu(self.conv1(x))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
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

		x = F.elu(x)
		#print(x.shape)
		x = x.view(-1, 64 * 1 * 18)
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		x = self.fc4(x)

		#print(x.cpu().detach().numpy())

		f = F.elu(f)
		#print(x.shape)
		f = f.view(-1, 64 * 1 * 18)
		f = F.elu(self.fc1(f))
		f = F.elu(self.fc2(f))
		f = F.elu(self.fc3(f))
		f = self.fc4(f)

		#print(f.cpu().detach().numpy())

		x = torch.cat((x,f),0)
		# print(out.shape)

		#print(x.shape)

		return x


from torch.autograd import Function
class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None




class net_nvidia_pytorch_DANN(nn.Module):
	# implementation of "Unsupervised Domain Adaptation by Backpropagation"
	def __init__(self):
		super(net_nvidia_pytorch_DANN, self).__init__()

		self.conv1 = nn.Conv2d(3, 24, 5, 2)
		self.conv2 = nn.Conv2d(24, 36, 5, 2)
		self.conv3 = nn.Conv2d(36, 48, 5, 2)
		self.conv4 = nn.Conv2d(48, 64, 3)
		self.conv5 = nn.Conv2d(64, 64, 3)

		self.fc1 = nn.Linear(64 * 1 * 18, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 10)
		self.fc4 = nn.Linear(10, 1)

		self.fc21 = nn.Linear(64 * 1 * 18, 100)
		self.fc22 = nn.Linear(100, 10)
		self.fc23 = nn.Linear(10, 2)

	def forward(self, input_data, alpha=1.0):
		x = LambdaLayer(lambda x: x/127.5 - 1.0)(input_data)
		x = F.elu(self.conv1(x))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
		x = F.elu(self.conv5(x))
		#print(x.shape)
		x = x.reshape(-1, 64 * 1 * 18)

		reverse_feature = ReverseLayerF.apply(x, alpha)

		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		regression_output = self.fc4(x)

		y = F.elu(self.fc1(reverse_feature))
		y = F.elu(self.fc2(y))
		domain_output = nn.LogSoftmax(dim=1)(self.fc3(y))

		return regression_output, domain_output