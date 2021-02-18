# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on ImageNet.

Currently only supports ResNet-50 training.

Example usage:
  `python imagenet.py <path/to/ImageNet> <path/to/ImageNet-C>`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import sys
import csv
import cv2
import math
cv2.setNumThreads(0)
# print(cv2.getNumThreads())
from PIL import Image

import augmentations

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from models.networks_pytorch import net_nvidia_pytorch, net_commaai_pytorch, net_resnet_pytorch

from sklearn.model_selection import train_test_split

augmentations.IMAGE_SIZE = 224

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__') and
                     callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier')
parser.add_argument(
    'clean_data', metavar='DIR', help='path to clean dataset')
parser.add_argument(
    'clean_data_label', metavar='LABEL_DIR', help='label path to clean dataset label')

parser.add_argument('--gpu_id', required=False, metavar="gpu_id", help='gpu id (0/1)')

# parser.add_argument(
#     'corrupted_data', metavar='DIR_C', help='path to ImageNet-C dataset')
parser.add_argument(
    '--model',
    '-m',
    default='nvidia_cnn',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet50)')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=4000, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0001,
    help='Weight decay (L2 penalty).')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=-1,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--aug-prob-coeff',
    default=1.,
    type=float,
    help='Probability distribution coefficients')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_false',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=10,
    help='Training loss print frequency (batches).')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=0,
    help='Number of pre-fetching threads.')

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
  b = args.batch_size / 256.
  k = args.epochs // 3
  if epoch < k:
    m = 1
  elif epoch < 2 * k:
    m = 0.1
  else:
    m = 0.01
  lr = args.learning_rate * m * b
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k."""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    ce = 100 * avg_err / ALEXNET_ERR[i]
    mce += ce / 15
  return mce


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return (torch.Tensor(image, dtype=torch.float32), torch.Tensor(labels, dtype=torch.float32))


class DrivingDataset_pytorch(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, xTrainList, yTrainList, transform=None, size=(200,66)):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.xTrainList = xTrainList
        self.yTrainList = yTrainList
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.yTrainList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.xTrainList[idx]

        image = cv2.imread(img_name)
        image = cv2.resize(image,self.size, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # image = Image.open(img_name)
        # image = image.resize((200, 66))
        # image = image.convert('YCbCr')

        labels = self.yTrainList[idx]
        labels = np.array([labels])
        labels = labels.astype('float32')
        sample = (image, labels)

        if self.transform:
            sample = self.transform(sample)

        return sample
        

def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(
      np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
  m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

  image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
  # mix = torch.zeros_like(preprocess(image))
  mix = np.zeros_like(image, dtype='float32')
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    # mix += ws[i] * preprocess(image_aug)
    mix += ws[i] * image_aug

  mixed = (1 - m) * image + m * mix
  mixed = cv2.cvtColor(mixed.astype('uint8'), cv2.COLOR_BGR2YUV)
  # mixed = transforms.ToTensor()(mixed)
  mixed = mixed.transpose((2, 0, 1))
  mixed = torch.tensor(mixed, dtype=torch.float32)
  # mixed = preprocess(mixed)
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=True, no_aug=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.no_aug = no_aug

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_aug:
      x = x.transpose((2, 0, 1))
      x = torch.tensor(x, dtype=torch.float32)
      return x, y
    elif self.no_jsd:
      return aug(x, self.preprocess), y

      # x = x.transpose((2, 0, 1))
      # x = torch.tensor(x, dtype=torch.float32)
      # return x, y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def load_train_data_multi(xFolder_list, trainLogPath_list, nRep = 1, fThreeCameras = False, ratio = 1.0, specialFilter = False):
  '''
  Load the training data
  '''
  ## prepare for getting x
  for xFolder in xFolder_list:
    if not os.path.exists(xFolder):
      sys.exit('Error: the image folder is missing. ' + xFolder)
    
  ## prepare for getting y
  trainLog_list = []
  for trainLogPath in trainLogPath_list:
    if not os.path.exists(trainLogPath):
      sys.exit('Error: the labels.csv is missing. ' + trainLogPath)
    with open(trainLogPath, newline='') as f:
      trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
      trainLog_list.append(trainLog)

  if not isinstance(ratio, list):
    ratio = [ratio]*len(xFolder_list)
  
    ## get x and y
  xList, yList = ([], [])
  
  for rep in range(0,nRep):
    i = 0
    for trainLog in trainLog_list:
      xFolder = xFolder_list[i]
      xList_1 = []
      yList_1 = []
      for row in trainLog:
        ## center camera
        if not specialFilter:
          xList_1.append(xFolder + os.path.basename(row[0])) 
          yList_1.append(float(row[3]))     
        elif float(row[3]) < 0:
          xList_1.append(xFolder + os.path.basename(row[0])) 
          yList_1.append(float(row[3]))
        
        ## if using three cameras
        if fThreeCameras:

          ## left camera
          xList_1.append(xFolder + row[1])  
          yList_1.append(float(row[3]) + 0.25) 
          
          ## right camera
          xList_1.append(xFolder + row[2])  
          yList_1.append(float(row[3]) - 0.25) 

      if ratio[i] < 1:
        n = int(len(trainLog) * ratio[i])

        #random.seed(42)
        #random.shuffle(xList_1)
        #random.seed(42)
        #random.shuffle(yList_1)
        xList_1, yList_1 = shuffle(xList_1, yList_1)

        xList_1 = xList_1[0:n]
        yList_1 = yList_1[0:n]
      print(len(xList_1))
      xList = xList + xList_1
      yList = yList + yList_1

      i+=1

  #yList = np.array(yList)*10 + 10
  return (xList, yList)
  

def train(net, train_loader, optimizer):
  """Train for one epoch."""
  net.train()
  data_ema = 0.
  batch_ema = 0.
  loss_ema = 0.
  acc1_ema = 0.
  acc5_ema = 0.

  start = time.time()
  end = time.time()
  # print(cv2.getNumThreads(), ' st')

  acc_list = [0,0,0,0,0,0]
  thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
  running_loss = 0
  epoch_time = 0
  data_time = 0
  criterion = nn.MSELoss()

  # print(cv2.getNumThreads())
  batch_num = len(train_loader)-1
  # batch_num = 2

  for i, (images, targets) in enumerate(train_loader):

    # image0 = images.detach().numpy()
    # for j in range(128):
    #   print(targets[j])
    #   image0_j = image0[j].transpose((1, 2, 0)).astype('uint8')
    #   cv2.imshow("img", image0_j)
    #   cv2.waitKey(0)

    # Compute data loading time
    data_time_1 = time.time() - end
    data_time += data_time_1
    optimizer.zero_grad()

    images = images.cuda()
    targets = targets.cuda()
    # print(images)
    logits,_ = net(images)
    # loss = F.mse_loss(logits, targets)
    loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()

    # print(logits[0])
    prediction_error = np.abs(logits.cpu().detach().numpy()-targets.cpu().detach().numpy())
    for j,thresh_hold in enumerate(thresh_holds):
      acc_count = np.sum(prediction_error < thresh_hold)
      acc_list[j] += acc_count

    # Compute batch computation time and update moving averages.
    # batch_time = time.time() - end
    end = time.time()

    # data_ema = data_ema * 0.1 + float(data_time) * 0.9
    # batch_time_ema = batch_ema * 0.1 + float(batch_time) * 0.9
    # loss_ema = loss_ema * 0.1 + float(loss) * 0.9
    # acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9

    running_loss += loss.item()
    if math.isnan(running_loss):
      print('image ============================================')
      print(images)
      print('logits ============================================')
      print(logits)
      print('targets ============================================')
      print(targets)
      break

    # if i % args.print_freq == 0:
    #   print(
    #       'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
    #       '{:.3f} | Train Acc5 {:.3f}'.format(i, len(train_loader), data_ema,
    #                                           batch_ema, loss_ema, acc1_ema,
    #                                           acc5_ema))

    if i >= batch_num -1:
      break

  epoch_time = time.time() - start

  acc = np.mean(acc_list) / batch_num / args.batch_size
  running_loss /= batch_num

  print('data_time ', data_time)

  return running_loss, acc, epoch_time


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  acc_list = [0,0,0,0,0,0]
  thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
  running_loss = 0
  batch_num = len(test_loader)-1
  # batch_num = 2

  with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
      images, targets = images.cuda(), targets.cuda()
      logits,_ = net(images)
      loss = F.mse_loss(logits, targets)
      running_loss += loss.item()

      prediction_error = np.abs(logits.cpu().detach().numpy()-targets.cpu().detach().numpy())
      for j,thresh_hold in enumerate(thresh_holds):
        acc_count = np.sum(prediction_error < thresh_hold)
        acc_list[j] += acc_count

      if i >= batch_num -1:
        break

  acc = np.mean(acc_list) / batch_num / args.batch_size
  running_loss /= batch_num

  return running_loss, acc


def test_c(net, test_transform):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = {}
  for c in CORRUPTIONS:
    print(c)
    for s in range(1, 6):
      valdir = os.path.join(args.corrupted_data, c, str(s))
      val_loader = torch.utils.data.DataLoader(
          datasets.ImageFolder(valdir, test_transform),
          batch_size=args.eval_batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          pin_memory=True)

      loss, acc1 = test(net, val_loader)
      if c in corruption_accs:
        corruption_accs[c].append(acc1)
      else:
        corruption_accs[c] = [acc1]

      print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(
          s, loss, 100. * acc1))

  return corruption_accs


def get_label_file_name(folder_name, suffix=""):
  pos = folder_name.find('_')
  if pos == -1:
    main_name = folder_name
  else:
    main_name = folder_name[0:pos]

  if "train" in folder_name:
    labelName = main_name.replace("train","labels") + "_train"
  elif "val" in folder_name:
    labelName = main_name.replace("val","labels") + "_val"

  labelName = labelName + suffix
  labelName = labelName + ".csv"
  return labelName


def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  # mean = [0.485, 0.456, 0.406]
  # std = [0.229, 0.224, 0.225]
  # train_transform = transforms.Compose(
  #     [transforms.RandomResizedCrop(224),
  #      transforms.RandomHorizontalFlip()])
  # preprocess = transforms.Compose(
  #     [transforms.ToTensor(),
  #      transforms.Normalize(mean, std)])
  preprocess = transforms.Compose(
      # [transforms.ToTensor(), transforms.resize(66, 200)]
      [transforms.ToTensor()]
      )
  # test_transform = transforms.Compose([
  #     transforms.Resize(256),
  #     transforms.CenterCrop(224),
  #     preprocess,
  # ])

  traindir = args.clean_data
  # valdir = traindir.replace('train', 'val')
  testdir = traindir.replace('train', 'val')
  label_file = args.clean_data_label

  xList, yList = load_train_data_multi([traindir], [label_file])
  xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
  yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)

  # BN_flag = 0 # nvidia net
  # BN_flag = 5 # comma.ai
  BN_flag = 8 # resnet

  size = (200, 66)
  if BN_flag == 8: #resnet
    size = (64, 64)

  train_dataset = DrivingDataset_pytorch(xTrainList, yTrainList, size=size)
  valid_dataset = DrivingDataset_pytorch(xValidList, yValidList, size=size)

  # train_dataset = datasets.ImageFolder(traindir, train_transform)
  train_dataset = AugMixDataset(train_dataset, preprocess)
  valid_dataset = AugMixDataset(valid_dataset, preprocess, no_aug=True)

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers)
  val_loader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers)

  # if args.pretrained:
  #   print("=> using pre-trained model '{}'".format(args.model))
  #   net = models.__dict__[args.model](pretrained=True)
  # else:
  #   print("=> creating model '{}'".format(args.model))
  #   net = models.__dict__[args.model]()

  if BN_flag == 0:
    net = net_nvidia_pytorch()
  elif BN_flag == 5:
    net = net_commaai_pytorch()
  elif BN_flag == 8:
    net = net_resnet_pytorch()

  print(net)

  # optimizer = torch.optim.SGD(
  #     net.parameters(),
  #     args.learning_rate,
  #     momentum=args.momentum,
  #     weight_decay=args.decay)

  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc1 = checkpoint['best_acc1']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  if args.evaluate:
    test_loss, test_acc1 = test(net, val_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Acc1 {:.3f}'.format(
        test_loss, 100 * test_acc1))

    corruption_accs = test_c(net, test_transform)
    for c in CORRUPTIONS:
      print('\t'.join([c] + map(str, corruption_accs[c])))

    print('mCE (normalized by AlexNet): ', compute_mce(corruption_accs))
    return

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          'imagenet_{}_training_log.csv'.format(args.model))
  with open(log_path, 'w') as f:
    f.write(
        'epoch,batch_time,train_loss,train_acc1(%),test_loss,test_acc1(%)\n')

  best_acc1 = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    # adjust_learning_rate(optimizer, epoch)

    train_loss, train_acc1, epoch_time = train(net, train_loader,
                                                      optimizer)
    test_loss, test_acc1 = test(net, val_loader)

    is_best = test_acc1 > best_acc1
    best_acc1 = max(test_acc1, best_acc1)
    checkpoint = {
        'epoch': epoch,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint_'+str(epoch)+'.pth')
    if (epoch % 50 == 0) or (epoch>=args.epochs-1):
      torch.save(checkpoint, save_path)
      if is_best:
        shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth'))

    with open(log_path, 'a') as f:
      f.write('%03d,%0.3f,%0.6f,%0.2f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          epoch_time,
          train_loss,
          100. * train_acc1,
          test_loss,
          100. * test_acc1,
      ))

    print(
        'Epoch {:3d} ({:.4f}) | Train Loss {:.4f} | Train Acc1 {:.2f} | Test Loss {:.3f} | Test Acc1 {:.2f}'
        .format((epoch + 1), epoch_time, train_loss, 100. * train_acc1, test_loss, 100. * test_acc1))

  # corruption_accs = test_c(net, test_transform)
  # for c in CORRUPTIONS:
  #   print('\t'.join(map(str, [c] + corruption_accs[c])))

  # print('mCE (normalized by AlexNet):', compute_mce(corruption_accs))


if __name__ == '__main__':
  if (args.gpu_id != None):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
  print("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])

  main()
