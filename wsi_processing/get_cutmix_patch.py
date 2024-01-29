from config import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import random
import pickle
import math
import cv2
import os

def get_clip_box(image_a, image_b):
    # image.shape = (height, width, channel)
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]

    # get center of box
    x = random.randrange(1,image_size_x+1)
    y = random.randrange(1,image_size_y+1)

    # get width, height of box
    width = int(512*math.sqrt(1-random.random()))
    height = int(512*math.sqrt(1-random.random()))

    # clip box in image and get minmax bbox
    xa = max(0, x-width//2)
    ya = max(0, y-height//2)
    xb = min(image_size_x, x+width//2)
    yb = min(image_size_y, y+width//2)
    return xa, ya, xb, yb

def mix_2_images(image_a, image_b, xa, ya, xb, yb):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0] 
    one = image_a[ya:yb,0:xa,:]
    two = image_b[ya:yb,xa:xb,:]
    three = image_a[ya:yb,xb:image_size_x,:]
    middle = np.concatenate([one,two,three],axis=1)
    top = image_a[0:ya,:,:]
    bottom = image_a[yb:image_size_y,:,:]
    mixed_img = np.concatenate([top, middle, bottom])
    return mixed_img

def mix_2_label(image_a, image_b, label_a, label_b, xa, ya, xb, yb):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0] 
    mixed_area = (xb-xa)*(yb-ya)
    total_area = image_size_x*image_size_y
    a = mixed_area/total_area

    mixed_label = (1-a)*label_a + a*label_b #
    return mixed_label[0]

def make_label(label):
    if label == 'normal': 
        return F.one_hot(torch.Tensor([0]).to(torch.int64), 4)
    elif label == 'leiomyoma':
        return F.one_hot(torch.Tensor([2]).to(torch.int64), 4)
    elif label == 'schwannoma':
        return F.one_hot(torch.Tensor([3]).to(torch.int64), 4)
        
def cutmix(a_image, b_image, a_label, b_label):
    xa, ya, xb, yb = get_clip_box(a_image, b_image)
    cutmix_image = mix_2_images(a_image, b_image, xa, ya, xb, yb)
    cutmix_label = mix_2_label(a_image, b_image, a_label, b_label, xa, ya, xb, yb)    
    return cutmix_image, cutmix_label    
    
def apply_augpatch(i, a_list, b_list, mixed_name, cutmix_tensor_zip):
    a_random = random.randrange(len(a_list))
    b_random = random.randrange(len(b_list))
    a_path   = a_list[a_random]
    b_path   = b_list[b_random]
    a_image  = cv2.imread(a_path)
    b_image  = cv2.imread(b_path)
    a, b     = mixed_name.split('_')
    a_label  = make_label(a)
    b_label  = make_label(b)

    cutmix_image, cutmix_label = cutmix(a_image, b_image, a_label, b_label)
    
    os.makedirs(f'{CUTMIX_PATH}/{mixed_name}', exist_ok=True)
    cutmix_path = f'{CUTMIX_PATH}/{mixed_name}/{i}.png'
    cv2.imwrite(cutmix_path, cutmix_image)
    cutmix_tensor_zip.append((cutmix_path, cutmix_label))
    
    return cutmix_tensor_zip


if __name__ == '__main__':

    # normal - schwannoma
    cutmix_tensor_zip1 = []
    for i in tqdm(range(NUM_NOR_SCH)):
        cutmix_tensor_zip1, = apply_augpatch(i, NORMAL_LIST, SCHWANNOMA_LIST, 'normal_schwannoma', cutmix_tensor_zip1)
    with open(f'{CUTMIX_PATH}/normal_schwannoma.pickle', 'wb') as fw:
        pickle.dump(cutmix_tensor_zip1, fw)

    # normal - leiomyoma
    cutmix_tensor_zip2 = []
    for i in tqdm(range(NUM_NOR_LEI)):
        cutmix_tensor_zip2 = apply_augpatch(i, NORMAL_LIST, LEIOMYOMA_LIST, 'normal_leiomyoma', cutmix_tensor_zip2)
    with open(f'{CUTMIX_PATH}/normal_leiomyoma.pickle', 'wb') as fw:
        pickle.dump(cutmix_tensor_zip2, fw)

    # leiomyoma - schwannoma
    cutmix_tensor_zip3 = []
    for i in tqdm(range(NUM_LEI_SCH)):
        cutmix_tensor_zip3 = apply_augpatch(i, LEIOMYOMA_LIST, SCHWANNOMA_LIST, 'leiomyoma_schwannoma', cutmix_tensor_zip3)
    with open(f'{CUTMIX_PATH}/leiomyoma_schwannoma.pickle', 'wb') as fw:
        pickle.dump(cutmix_tensor_zip3, fw)