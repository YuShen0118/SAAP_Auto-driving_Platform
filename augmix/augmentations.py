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
"""Base augmentations operators."""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
import cv2
cv2.setNumThreads(0)

# ImageNet code should change this value
IMAGE_SIZE = 32


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


# augmentations = [
#     autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y
# ]

# augmentations_all = [
#     autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
#     translate_x, translate_y, color, contrast, brightness, sharpness
# ]

RATIO_LIST = [0.02, 0.2, 0.5, 0.65, 1.0]
RGB_MAX = 255
HSV_H_MAX = 180
HSV_SV_MAX = 255

def change_channel_value(image, channel, level, maxval):
  if level < 0:
    level = random.randint(0, 9)
  # 0~4 darker, 5~9 lighter
  if level <= 4:
    dist_ratio = RATIO_LIST[4 - level]
    image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio))
  else:
    dist_ratio = RATIO_LIST[level - 5]
    image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (maxval * dist_ratio)

def R_channel(image, level):
  # 0~4 darker, 5~9 lighter
  # print('R_channel')
  channel = 2
  # image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
  change_channel_value(image, channel, level, RGB_MAX)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
  return image

def G_channel(image, level):
  # 0~4 darker, 5~9 lighter
  # print('G_channel')
  channel = 1
  # image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
  change_channel_value(image, channel, level, RGB_MAX)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
  return image

def B_channel(image, level):
  # 0~4 darker, 5~9 lighter
  # print('B_channel')
  channel = 0
  # image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
  change_channel_value(image, channel, level, RGB_MAX)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
  return image

def H_channel(image, level):
  # 0~4 darker, 5~9 lighter
  # print('H_channel')
  channel = 0
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  change_channel_value(image, channel, level, HSV_H_MAX)
  image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
  return image

def S_channel(image, level):
  # 0~4 darker, 5~9 lighter
  # print('S_channel')
  channel = 1
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  change_channel_value(image, channel, level, HSV_SV_MAX)
  image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
  return image

def V_channel(image, level):
  # 0~4 darker, 5~9 lighter
  # print('V_channel')
  channel = 2
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  change_channel_value(image, channel, level, HSV_SV_MAX)
  image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
  return image


BLUR_LVL = [7, 17, 37, 67, 107]

def Gaussian_blur(image, level):
  # print('Gaussian_blur')
  if level < 0:
    level = random.randint(0, 4)
  kernal_size = BLUR_LVL[level]
  image = cv2.GaussianBlur(image, (kernal_size, kernal_size), 0).astype('uint8')
  return image


NOISE_LVL = [20, 50, 100, 150, 200]

def add_noise(image, sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = (image + gauss).astype('uint8')
    return noisy

def Gaussian_noise(image, level):
  # print('Gaussian_noise')
  if level < 0:
    level = random.randint(0, 4)
  sigma = NOISE_LVL[level]
  image = add_noise(image, sigma)
  return image


DIST_LVL = [1, 10, 50, 200, 500]

def Distortion(image, level):
  # print('Distortion')
  if level < 0:
    level = random.randint(0, 4)
  K = np.eye(3)*1000
  K[0,2] = image.shape[1]/2
  K[1,2] = image.shape[0]/2
  K[2,2] = 1

  k1 = DIST_LVL[level]
  k2 = DIST_LVL[level]
  image = (cv2.undistort(image, K, np.array([k1,k2,0,0]))*255.0).astype('uint8')
  return image


augmentations = [
    R_channel, G_channel, B_channel, 
    H_channel, S_channel, V_channel,
    Gaussian_blur, Gaussian_noise, Distortion
]

augmentations_all = [
    R_channel, G_channel, B_channel, 
    H_channel, S_channel, V_channel,
    Gaussian_blur, Gaussian_noise, Distortion
]