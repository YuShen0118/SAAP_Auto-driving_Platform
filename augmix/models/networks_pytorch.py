import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


def create_nvidia_network_pytorch(BN_flag, fClassifier=False, nClass=1, nChannel=3, Maxup_flag=False):
	#default
	if BN_flag == 0:
		return net_nvidia_pytorch(nChannel)
	elif BN_flag == 3:
		return net_nvidia_featshift_pytorch()
	elif BN_flag == 4:
		return net_nvidia_pytorch_DANN()
	elif BN_flag == 5:
		return net_commaai_pytorch()
	elif BN_flag == 7:
		return net_nvidia_pytorch_LSTM(nChannel)
	elif BN_flag == 8:
		return net_resnet_pytorch(nChannel)
		
	return net_nvidia_pytorch()

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class net_nvidia_pytorch(nn.Module):
	def __init__(self, nChannel=3):
		super(net_nvidia_pytorch, self).__init__()
		self.conv1 = nn.Conv2d(nChannel, 24, 5, 2)
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
		last_layer_feature = F.elu(self.fc3(x))
		output = self.fc4(last_layer_feature)
		return output, last_layer_feature

class net_commaai_pytorch(nn.Module):
	def __init__(self):
		super(net_commaai_pytorch, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 8, 4, padding=8)
		self.conv2 = nn.Conv2d(16, 32, 5, 2, padding=5)
		self.conv3 = nn.Conv2d(32, 64, 5, 2, padding=5)
		self.fc1 = nn.Linear(64 * 10 * 18, 512)
		self.fc2 = nn.Linear(512, 1)

	def forward(self, x):
		x = LambdaLayer(lambda x: x/127.5 - 1.0)(x)
		x = F.elu(self.conv1(x))
		x = F.elu(self.conv2(x))
		x = self.conv3(x)
		x = x.reshape(-1, 64 * 10 * 18)
		x = F.elu(F.dropout(x, .2))
		#print(x.shape)
		# x = x.reshape(-1, 64 * 1 * 18)
		x = self.fc1(x)
		x = F.elu(F.dropout(x, .5))
		x = self.fc2(x)
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



# ADDA models
class ADDA_NVIDIA_FEATURE_CNN(nn.Module):
	# implementation of "Unsupervised Domain Adaptation by Backpropagation"
	def __init__(self, nChannel=3):
		super(ADDA_NVIDIA_FEATURE_CNN, self).__init__()

		self.conv1 = nn.Conv2d(nChannel, 24, 5, 2)
		self.conv2 = nn.Conv2d(24, 36, 5, 2)
		self.conv3 = nn.Conv2d(36, 48, 5, 2)
		self.conv4 = nn.Conv2d(48, 64, 3)
		self.conv5 = nn.Conv2d(64, 64, 3)
		self.fc1 = nn.Linear(64 * 1 * 18, 100)

	def forward(self, input_data, alpha=1.0):
		# x = LambdaLayer(lambda x: x/127.5 - 1.0)(input_data)
		x = F.elu(self.conv1(input_data))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
		x = F.elu(self.conv5(x))
		#print(x.shape)
		x = x.reshape(-1, 64 * 1 * 18)
		feature = F.elu(self.fc1(x))

		return feature


class ADDA_NVIDIA_REGRESSOR(nn.Module):
	# implementation of "Unsupervised Domain Adaptation by Backpropagation"
	def __init__(self):
		super(ADDA_NVIDIA_REGRESSOR, self).__init__()

		self.fc1 = nn.Linear(100, 50)
		self.fc2 = nn.Linear(50, 10)
		self.fc3 = nn.Linear(10, 1)

	def forward(self, input_data):

		x = F.elu(self.fc1(input_data))
		x = F.elu(self.fc2(x))
		regression_output = self.fc3(x)

		return regression_output


class ADDA_DOMAIN_DISCRIMINATOR(nn.Module):
	# implementation of "Unsupervised Domain Adaptation by Backpropagation"
	def __init__(self):
		super(ADDA_DOMAIN_DISCRIMINATOR, self).__init__()

		self.fc1 = nn.Linear(100, 10)
		self.fc2 = nn.Linear(10, 2)

	def forward(self, input_data):
		
		y = F.elu(self.fc1(input_data))
		domain_output = nn.LogSoftmax(dim=1)(self.fc2(y))

		return domain_output


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        if (len(x.shape)==3):
            x_reshape = x.contiguous().view(x.shape[0]*x.shape[1], x.shape[2])  # (samples * timesteps, input_size)
        elif (len(x.shape)==5):
            x_reshape = x.contiguous().view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            if type(y) is tuple:
                y1, y2 = y
                y1 = y1.contiguous().view(x.size(0), -1, y1.size(-1))  # (samples, timesteps, output_size)
                y2 = y2.contiguous().view(x.size(0), -1, y2.size(-1))  # (samples, timesteps, output_size)
            else:
                y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            if type(y) is tuple:
                y1, y2 = y
                y1 = y1.view(-1, x.size(1), y1.size(-1))  # (timesteps, samples, output_size)
                y2 = y2.view(-1, x.size(1), y2.size(-1))  # (timesteps, samples, output_size)
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class net_nvidia_pytorch_CNN(nn.Module):
	# implementation of "Unsupervised Domain Adaptation by Backpropagation"
	def __init__(self, nChannel=3):
		super(net_nvidia_pytorch_CNN, self).__init__()

		self.conv1 = nn.Conv2d(nChannel, 24, 5, 2)
		self.conv2 = nn.Conv2d(24, 36, 5, 2)
		self.conv3 = nn.Conv2d(36, 48, 5, 2)
		self.conv4 = nn.Conv2d(48, 64, 3)
		self.conv5 = nn.Conv2d(64, 64, 3)

	def forward(self, input_data):
		# x = LambdaLayer(lambda x: x/127.5 - 1.0)(input_data)
		x = F.elu(self.conv1(input_data))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
		x = F.elu(self.conv5(x))
		#print(x.shape)
		output = x.reshape(-1, 64 * 1 * 18)

		return output


class net_nvidia_pytorch_regressor(nn.Module):
	# implementation of "Unsupervised Domain Adaptation by Backpropagation"
	def __init__(self):
		super(net_nvidia_pytorch_regressor, self).__init__()

		self.fc1 = nn.Linear(64 * 1 * 18, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 10)
		self.fc4 = nn.Linear(10, 1)

	def forward(self, input_data):

		x = F.elu(self.fc1(input_data))
		x = F.elu(self.fc2(x))
		last_layer_feature = F.elu(self.fc3(x))
		regression_output = self.fc4(last_layer_feature)

		return regression_output, last_layer_feature


class net_nvidia_pytorch_LSTM(nn.Module):
	def __init__(self, nChannel=3):
		super(net_nvidia_pytorch_LSTM, self).__init__()
		self.cnn = TimeDistributed(net_nvidia_pytorch_CNN(nChannel), batch_first=True)
		self.lstm = nn.LSTM(64 * 1 * 18, 64 * 1 * 18, 2, batch_first=True)
		# self.lstm = nn.LSTM(3 * 66 * 200, 3 * 66 * 200, 2, batch_first=True)
		self.regressor = TimeDistributed(net_nvidia_pytorch_regressor(), batch_first=True)

	def forward(self, x):
		x = self.cnn(x)
		x,_ = self.lstm(x)
		regression_output, last_layer_feature = self.regressor(x)
		return regression_output, last_layer_feature


class net_resnet_pytorch(nn.Module):
	def __init__(self, nChannel=3):
		super(net_resnet_pytorch, self).__init__()
		self.conv1 = nn.Conv2d(nChannel, 3, 1)
		self.resnet152 = models.resnet152()
		self.header = nn.Linear(1000, 1)

	def forward(self, x):
		x = LambdaLayer(lambda x: x/127.5 - 1.0)(x)
		x = self.conv1(x)
		last_layer_feature = self.resnet152(x)
		output = self.header(last_layer_feature)
		return output, last_layer_feature