from config                                  import *
from skimage                                 import morphology, filters
from tqdm                                    import trange
import numpy                                 as np
import openslide
import cv2
import os


class PatchProcessor(object):
    def __init__(self, patch_size, resolution, down_level):
        self.patch_size = patch_size
        self.resolution = resolution
        self.down_level = down_level

    def _slide_setting(self, slide_path, anno_path, save_path):
        self.slide_path = slide_path
        self.anno_path  = anno_path
        self.save_path  = save_path

        self.slide = openslide.OpenSlide(self.slide_path)
        self.slide_name = '.'.join(self.slide_path.split('/')[-1].split('.')[:-1])
        self.slide_type = self.slide_path.split('/')[-2]
        self.slide_organ= self.slide_path.split('/')[-3]

        self.level0_w, self.level0_h = self.slide.level_dimensions[0]
        self.adjust_term = 1 / float(self.slide.properties.get('openslide.mpp-x'))
        self.read_size = int(self.patch_size*self.adjust_term)

        self.level_min_w, self.level_min_h = self.slide.level_dimensions[self.down_level]
        self.level_min_img = np.array(self.slide.read_region((0,0), self.down_level if self.down_level > 0 else self.slide.level_count-1, size=(self.level_min_w, self.level_min_h)))[...,:3]
        self.zero2min = self.level0_h // self.level_min_h

    # get tissue mask method
    def get_tissue_mask(self, RGB_min=0):
        level_min_w, level_min_h = self.slide.level_dimensions[-1]
        level_min_img = np.array(self.slide.read_region((0,0), self.slide.level_count-1, size=(level_min_w, level_min_h)))
        hsv = cv2.cvtColor(level_min_img, cv2.COLOR_RGB2HSV)
        ## if more than threshold make Ture
        background_R = level_min_img[:, :, 0] > filters.threshold_otsu(level_min_img[:, :, 0])
        background_G = level_min_img[:, :, 1] > filters.threshold_otsu(level_min_img[:, :, 1])
        background_B = level_min_img[:, :, 2] > filters.threshold_otsu(level_min_img[:, :, 2])

        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])

        min_R = level_min_img[:, :, 0] > RGB_min
        min_G = level_min_img[:, :, 1] > RGB_min
        min_B = level_min_img[:, :, 2] > RGB_min

        mask = tissue_S & (tissue_RGB + min_R + min_G + min_B)
        ret = morphology.remove_small_holes(mask, area_threshold=(level_min_h*level_min_w)//8)
        ret = np.array(ret).astype(np.uint8)
        
        kernel_size = 5
        #kernel_size = 10*self.zero2min
        tissue_mask = cv2.morphologyEx(ret*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size)))  
        tissue_mask = cv2.resize(tissue_mask, (self.level_min_w, self.level_min_h), cv2.INTER_NEAREST)
        return tissue_mask

    # get sequence ratio method
    def get_seq_range(self, slide_width, slide_height, read_size, zero2two):
        y_seq = trange(int(((slide_height)) // int(read_size/zero2two)) + 1)
        x_seq = range(int(((slide_width)) // int(read_size/zero2two)) + 1)
        return y_seq, x_seq

    # get ratio mask method
    def get_ratio_mask(self, patch):
        h_, w_ = patch.shape[0], patch.shape[1]
        n_total = h_*w_
        n_cell = np.count_nonzero(patch)
        if (n_cell != 0):
            return n_cell*1.0/n_total*1.0
        else:
            return 0

    # save patch method
    def save_image(self, dir_path, file_name, img):
        os.makedirs(dir_path, exist_ok = True)
        cv2.imwrite(os.path.join(dir_path, file_name), img)


class ExtractPatch(PatchProcessor):

    # init variable setting method
    def __init__(self, patch_size, resolution, down_level, anno_ratio, tissue_ratio):
        super().__init__(
            patch_size  = patch_size,
            resolution  = resolution,
            down_level  = down_level           
        )
        self.anno_ratio = anno_ratio
        self.tissue_ratio  = tissue_ratio

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    # extract patch from get patch 
    def execute_patch(self, patch_img, patch_count, name):
        resize_image = cv2.resize(patch_img, (self.patch_size,self.patch_size), cv2.INTER_AREA)
        self.save_image(self.save_path + f'/{self.slide_organ}/{self.slide_type}/{self.slide_name}', f'{patch_count}_{name}.png', resize_image)


    def except_black(self):
        check_patch = np.array(self.slide.read_region((0,0), 0, size=(self.read_size, self.read_size)))[...,:3]
        try:
            min_x = int(np.where(check_patch>[0,0,0])[0][0]/self.zero2min)+1
            min_y = int(np.where(check_patch>[0,0,0])[1][0]/self.zero2min)+1
        except:
            min_x = 1
            min_y = 1
        return min_x, min_y

    # extract patches corresponding with annoation mask method
    def extract(self):
        tissue_mask = self.get_tissue_mask()
        step = 1
        patch_count = 0
        blur_count = 0

        min_x, min_y = self.except_black() 
        slide_w=self.level_min_w; slide_h=self.level_min_h
        y_seq, x_seq = self.get_seq_range(slide_w, slide_h, self.read_size, self.zero2min)

        for y in y_seq:
            for x in x_seq:
                start_x = int(min_x + int(self.read_size/self.zero2min)*x)
                end_x = int(min_x + int(self.read_size/self.zero2min)*(x+step))
                start_y = int(min_y + int(self.read_size/self.zero2min)*y)
                end_y = int(min_y+ int(self.read_size/self.zero2min)*(y+step))

                tissue_mask_patch = tissue_mask[start_y:end_y, start_x:end_x]
                if (self.get_ratio_mask(tissue_mask_patch) >= self.tissue_ratio):
                
                    img_patch = np.array(self.slide.read_region(
                        location = (int(start_x*self.zero2min), int(start_y*self.zero2min)),
                        level = 0,
                        size = (self.read_size, self.read_size)
                    )).astype(np.uint8)[...,:3]

                    if self.variance_of_laplacian(img_patch) >= 100:
                        patch_count += 1 
                        name = 'unknown'

                        self.execute_patch(img_patch, patch_count, name)
                    else:
                        blur_count += 1
                        self.save_image(self.save_path + f'/blur/{self.slide_organ}/{self.slide_type}/{self.slide_name}', f'{blur_count}.png', img_patch)