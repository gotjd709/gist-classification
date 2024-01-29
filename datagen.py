from albumentations.pytorch                   import ToTensorV2
from sklearn.utils                            import shuffle
from torch.utils.data                         import Dataset
from torchsampler                             import ImbalancedDatasetSampler
from config                                   import *
import albumentations                         as A
import numpy                                  as np
import pickle
import torch
import cv2


class TensorData(Dataset):    
    def __init__(self, data_zip, augmentation=None):
        self.image_path    = [x[0] for x in data_zip]
        self.label         = [x[1] for x in data_zip]
        self.augmentation  = train_aug() if augmentation else test_aug()

    def get_labels(self):
        return [torch.argmax(x).item() for x in self.label]
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, index):
        img_path = self.image_path[index]
        batch_x = cv2.imread(img_path).astype(np.float32)/255
        sample = self.augmentation(image=batch_x)
        x_data = sample['image']
        y_data = self.label[index].to(torch.float32)
        return x_data, y_data


def dataloader_setting(cutmix, imb_sampler):
    # Path Setting    
    with open(TRAIN_PATH, 'rb') as fr:
        train_zip = pickle.load(fr)
    with open(VALID_PATH, 'rb') as fr:
        valid_zip = pickle.load(fr)
    with open(TEST_PATH, 'rb') as fr:
        test_zip = pickle.load(fr) 

    if cutmix:
        with open(NOR_SCH_PATH) as fr:
            n_s = pickle.load(fr)
        with open(NOR_LEI_PATH) as fr:
            n_l = pickle.load(fr)
        with open(LEI_SCH_PATH) as fr:
            l_s = pickle.load(fr) 
        train_zip.extend(n_s)
        train_zip.extend(n_l)
        train_zip.extend(l_s)
    else:
        pass
    
    train_zip = shuffle(train_zip)
    train_data = TensorData(train_zip, augmentation=True)
    valid_data = TensorData(valid_zip)
    test_data = TensorData(test_zip)
    
    sampler = ImbalancedDatasetSampler(train_data) if imb_sampler else None
    # DataLoader Setting
    train_loader = torch.utils.data.DataLoader(
        dataset         = train_data,
        sampler         = sampler,
        batch_size      = BATCH_SIZE,
        shuffle         = False if imb_sampler else True,
        num_workers     = NUM_WORKER,
        pin_memory      = True,
        prefetch_factor = 2*NUM_WORKER,
        drop_last       = True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset         = valid_data,
        batch_size      = BATCH_SIZE,
        shuffle         = False,
        num_workers     = NUM_WORKER,
        pin_memory      = True,
        prefetch_factor = 2*NUM_WORKER,
        drop_last       = True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset         = test_data,
        batch_size      = BATCH_SIZE,
        shuffle         = False,
        num_workers     = NUM_WORKER,
        pin_memory      = True,
        prefetch_factor = 2*NUM_WORKER,
        drop_last       = True
    )
    return train_loader, valid_loader, test_loader


def train_aug():
    ret = A.Compose(
        [
            A.OneOf([
                A.ShiftScaleRotate(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ],p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit = (-0.4,0.4),
                    contrast_limit   = (-0.4,0.4)
                ),
                A.HueSaturationValue(
                    hue_shift_limit = (-30,30),
                    sat_shift_limit = (-40,40),
                    val_shift_limit = (-30,30)
                )
            ],p=0.9),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=20, alpha_affine=15, interpolation=1, border_mode=1),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ],p=0.5),
            ToTensorV2()
        ]
    )
    return ret

def test_aug():
    ret = A.Compose(
        [
            ToTensorV2()
        ]   
    )
    return ret