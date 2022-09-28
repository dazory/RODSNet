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
import torch
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from torchvision import datasets

from torchvision.transforms.functional import to_pil_image

# ImageNet code should change this value
#IMAGE_SIZE = 32

#########################################################
#################### AUGMENTATIONS ######################
#########################################################


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


def autocontrast(pil_img, _, __):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _, __):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level, _):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, _):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, _):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, image_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(image_size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level, image_size):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(image_size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level, image_size):
  level = int_parameter(sample_level(level), image_size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(image_size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level, image_size):
  level = int_parameter(sample_level(level), image_size[1] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(image_size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

"""
pixmix_no_translate = [
autocontrast, equalize, posterize, solarize
]


pixmix = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

pixmix_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]
"""

#########################################################
######################## MIXINGS ########################
#########################################################

def get_ab(beta):
  if np.random.random() < 0.5:
    a = np.float32(np.random.beta(beta, 1))
    b = np.float32(np.random.beta(1, beta))
  else:
    a = 1 + np.float32(np.random.beta(1, beta))
    b = -np.float32(np.random.beta(1, beta))
  return a, b

def add(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 1, img2 * 2 - 1
  out = a * img1 + b * img2
  return (out + 1) / 2

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clamp(1e-37) ** b)
  return out / 2




########################################
##### EXTRA MIXIMGS (EXPREIMENTAL) #####
########################################

def invert(img):
  return 1 - img

def screen(img1, img2, beta):
  img1, img2 = invert(img1), invert(img2)
  out = multiply(img1, img2, beta)
  return invert(out)

def overlay(img1, img2, beta):
  case1 = multiply(img1, img2, beta)
  case2 = screen(img1, img2, beta)
  if np.random.random() < 0.5:
    cond = img1 < 0.5
  else:
    cond = img1 > 0.5
  return torch.where(cond, case1, case2)

def darken_or_lighten(img1, img2, beta):
  if np.random.random() < 0.5:
    cond = img1 < img2
  else:
    cond = img1 > img2
  return torch.where(cond, img1, img2)

def swap_channel(img1, img2, beta):
  channel = np.random.randint(3)
  img1[channel] = img2[channel]
  return img1



#to_tensor = transforms.ToTensor()
#normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
"""/////////////////////////////////////////////////////////"""

class PixMix:
    def __init__(self, mixing_set, pixmix_auglist, pixmix_method, normalization):
        self.aug_severity = 3
        self.k = 4
        self.beta = 3
        self.pixmix_auglist = pixmix_auglist
        self.mixing_set = mixing_set
        self.pixmix_method = pixmix_method
        self.normalize = normalization
        self.pixmix_no_translate = [
            autocontrast, equalize, posterize, solarize
        ]
        self.pixmix = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y
        ]
        self.pixmix_all = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y, color, contrast, brightness, sharpness
        ]


    def __call__(self, results):
        img_left = results['left'].copy()
        img_right = results['right'].copy()



        mixing_set = datasets.ImageFolder(root = self.mixing_set, transform = transforms.Resize((img_left.height, img_left.width)))
        rnd_idx = np.random.choice(len(mixing_set))
        mixing_pic, _ = mixing_set[rnd_idx]


        if self.pixmix_method == 'no_jsd':
            results['left'] = self.pixmix_op(img_left, mixing_pic, self.normalize)
            results['right'] = self.pixmix_op(img_right, mixing_pic, self.normalize)
            return results



        elif self.pixmix_method == 'copy':
            results['left'] = transforms.ToTensor()(img_left)
            results['right'] = transforms.ToTensor()(img_right)

            if self.normalize == 1:
                results['left'] = transforms.Normalize((0.28689554, 0.32513303, 0.28389177), (0.18696375, 0.19017339, 0.18720214))(results['left'])
                results['right'] = transforms.Normalize((0.28689554, 0.32513303, 0.28389177), (0.18696375, 0.19017339, 0.18720214))(results['right'])

            results['left_aug1'] = results['left'].copy()
            results['left_aug2'] = results['left'].copy()
            results['right_aug1'] = results['right'].copy()
            results['right_aug2'] = results['right'].copy()

            return results


        else:
            results['left'] = transforms.ToTensor()(img_left)
            results['right'] = transforms.ToTensor()(img_right)
            results['left_aug1'] = self.pixmix_op(img_left, mixing_pic, self.normalize)
            results['left_aug2'] = self.pixmix_op(img_left, mixing_pic, self.normalize)
            results['right_aug1'] = self.pixmix_op(img_right, mixing_pic, self.normalize)
            results['right_aug2'] = self.pixmix_op(img_right, mixing_pic, self.normalize)

            if self.normalize == 1:
                results['left'] = transforms.Normalize((0.28689554, 0.32513303, 0.28389177), (0.18696375, 0.19017339, 0.18720214))(results['left'])
                results['right'] = transforms.Normalize((0.28689554, 0.32513303, 0.28389177), (0.18696375, 0.19017339, 0.18720214))(results['right'])

            return results


    def pixmix_op(self, orig, mixing_pic, normalize):
        mixings = [add, multiply]
        #tensorize, normalize = preprocess['tensorize'], preprocess['normalize']

        if np.random.random() < 0.5:
            mixed = transforms.ToTensor()(self.augment_input(orig))
        else:
            mixed = transforms.ToTensor()(orig)

        for _ in range(np.random.randint(self.k + 1)):

            if np.random.random() < 0.5:
                aug_image_copy = transforms.ToTensor()(self.augment_input(orig))
            else:
                aug_image_copy = transforms.ToTensor()(mixing_pic)


            mixed_op = np.random.choice(mixings)
            mixed = mixed_op(mixed, aug_image_copy, self.beta)
            mixed = torch.clamp(mixed, 0, 1)


        if normalize == 1:
            mixed = transforms.Normalize((0.28689554,0.32513303,0.28389177),(0.18696375,0.19017339,0.18720214))(mixed)


        return mixed


    def augment_input(self, orig):
        #pixmix_auglist = utils.augmentations_all if args.all_ops else utils.augmentations
        img_size = orig.size
        if self.pixmix_auglist == 'pixmix_no_translate':
            pixmix_auglist = self.pixmix_no_translate
        elif self.pixmix_auglist == 'pixmix_basic':
            pixmix_auglist = self.pixmix
        else :
            pixmix_auglist = self.pixmix_all

        op = np.random.choice(pixmix_auglist)
        return op(orig.copy(), self.aug_severity, img_size)
