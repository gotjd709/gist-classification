from albumentations.pytorch                  import ToTensorV2
from openslide.deepzoom                      import DeepZoomGenerator
from torch.utils.data                        import DataLoader, Dataset
from config                                  import *
from skimage                                 import filters
from tqdm                                    import tqdm
import albumentations                        as A
import torch.nn                              as nn
import pandas                                as pd
import numpy                                 as np
import openslide
import torch
import cv2
import os

class SetWSI(Dataset):
    def __init__(self, slide_path, target_mpp, target_size, overlap=0, return_loc=False, shuffle=False, mode='inference'):
        self.slide_path      = slide_path
        self.slide_name      = '.'.join(os.path.split(self.slide_path)[-1].split('.')[:-1])
        self.target_mpp      = target_mpp
        self.target_size     = target_size
        self.slide           = openslide.open_slide(self.slide_path)
        self.overlap         = 1-overlap
        self.dzi_size        = int(self.target_size*self.overlap)
        self.dzi_overlap     = int((self.target_size-self.dzi_size)//2)
        self.dzi             = DeepZoomGenerator(self.slide,tile_size=self.dzi_size,overlap=self.dzi_overlap)
        self.rgb_min         = 255*.25
        self.rgb_max         = 255*.9
        self.transform       = A.Compose([A.ToFloat(),ToTensorV2()])
        self.shuffle         = shuffle
        self.return_loc      = return_loc
        self.mode = mode
        self._init_property()
        self._init_thumbnail()
        self._init_grid()
    
    def _init_property(self):
        try:
            mpp = float(f'{float(self.slide.properties.get("openslide.mpp-x")):.2f}')
        except:
            mpp = .25
        self.target_downsample = int(self.target_mpp/mpp)
        self.target_dim = tuple(x//self.target_downsample for x in self.slide.level_dimensions[0])

        return self.target_downsample
    
    @staticmethod
    def get_tissue_mask(rgb_image, morph=None, morph_kernel=(5, 5)):
        hsv = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)
        background_R = rgb_image[:, :, 0] > filters.threshold_otsu(rgb_image[:, :, 0])
        background_G = rgb_image[:, :, 1] > filters.threshold_otsu(rgb_image[:, :, 1])
        background_B = rgb_image[:, :, 2] > filters.threshold_otsu(rgb_image[:, :, 2])

        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])

        min_R = rgb_image[:, :, 0] > 0
        min_G = rgb_image[:, :, 1] > 0
        min_B = rgb_image[:, :, 2] > 0

        mask = tissue_S & (tissue_RGB + min_R + min_G + min_B)
        ret = mask
        ret = np.array(ret).astype(np.uint8)
        if morph == 'open':
            ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel))
        elif morph == 'close':
            ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel))
        else:
            ret = ret
        # 5% kernel을 morphology연산으로 사용
        return ret

    @staticmethod
    def get_rgb_val(image, location):
        y_s = location[0]; x_s = location[1]
        return np.mean(image[y_s:y_s+1, x_s:x_s+1])

    def _init_thumbnail(self):
        self.thumbnail = np.array(self.slide.get_thumbnail(
            (self.target_dim[0]//(self.dzi_size), self.target_dim[1]/(self.dzi_size))
        ).convert("RGB"))
        self.thumbnail_mask = self.get_tissue_mask(self.thumbnail)


    def _init_grid(self):
        binary = self.thumbnail_mask>0
        try:
            h,w = self.thumbnail_mask.shape
            for i in range(len(self.dzi.level_tiles)):
                _, tile_h = self.dzi.level_tiles[i]
                if (h/tile_h) > .85 and (h/tile_h) < 1.15:
                    self.dzi_lv = i
        except:
            self.dzi_lv = -1

        self.df = pd.DataFrame(pd.DataFrame(binary).stack())
        self.df['is_tissue'] = self.df[0]; self.df.drop(0, axis=1, inplace=True)
        self.df['slide_path'] = self.slide_path
        self.df.query('is_tissue==True', inplace=True)
        index_list = list(self.df.index)
        rm_list = []
        for i,index in enumerate(index_list):
            if index[0] <= 2 or index[1] <= 2: # rm first column or first row for avoiding collate error caused tile size
                rm_list.append(index)
            if index[0] >= h-2 or index[1] >= w-2: # rm last column or last row for avoiding collate error caused tile size
                rm_list.append(index)
        rm_list = list(tuple(rm_list))
        self.df['tile_loc'] = index_list
        self.df['rgb'] = self.df.apply(lambda x : self.get_rgb_val(self.thumbnail, x['tile_loc']), axis=1)
        self.df.query(f'rgb>{self.rgb_min} & rgb<{self.rgb_max}', inplace=True)
        [self.df.drop(x, inplace=True) for x in rm_list if x in self.df.index]
        self.df.reset_index(inplace=True, drop=True)
        
        if self.shuffle==True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.tile_loc_list = [x[::-1] for x in self.df.tile_loc]
                

    def __len__(self):
        return len(self.tile_loc_list)


    def __getitem__(self, idx):
        tile_loc = self.tile_loc_list[idx]
        tile = np.array(self.dzi.get_tile(self.dzi_lv, tile_loc).convert("RGB")).astype(np.uint8)
        tile = self.transform(image=tile)['image']
        
        if self.return_loc is True:
            return (tile, str(tile_loc))
        
        else:
            return tile

class InferenceWSI(object):
    def __init__(self, model_path, target_mpp, target_size, overlap, slide_organ, save_base_dir, batch_size=32):
        self.model          = torch.load(model_path)
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target_mpp     = target_mpp
        self.target_size    = target_size
        self.overlap        = overlap
        self.add_pix        = int(1/(1-self.overlap))
        self.batch_size     = batch_size
        self.softmax        = nn.Softmax() 
        self.slide_organ    = slide_organ
        self.save_base_dir  = save_base_dir
        
    def image_with_mask(self, image, mask):
        image_orig = image.copy()
        color_dict = {1:(255,0,0), 2:(0,0,255), 3:(44,44,44)}
        for i in range(3):
            mask_type    = np.where(mask==i+1,1,0).astype(np.uint8)
            mask_cont, _ = cv2.findContours(mask_type, cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE) 
            image        = cv2.fillPoly(image, mask_cont, color_dict[i+1])
        img_with_msk = cv2.addWeighted(src1=image_orig, alpha=0.6, src2=image, beta=0.4, gamma=0)
        return img_with_msk

    def get_predict(self, slide_path, total_df, csv_save_path):
        slide       = openslide.open_slide(slide_path)
        slide_image = np.array(slide.get_thumbnail((int(slide.level_dimensions[1][0]),int(slide.level_dimensions[1][1]))))[...,:3]
        slide_name  = '.'.join(slide_path.split('/')[-1].split('.')[:-1])
        wsi_set     = SetWSI(slide_path, self.target_mpp, self.target_size, overlap=self.overlap, return_loc=True)
        wsi_loader  = DataLoader(wsi_set, batch_size=self.batch_size)

        ret0 = np.zeros((wsi_set.thumbnail_mask.shape[0]*self.target_size, wsi_set.thumbnail_mask.shape[1]*self.target_size))
        loc_x_list = []; loc_y_list = []; preds_list = []; probs_list = []
        
        for i, (batch_tiles, batch_tile_loc) in tqdm(enumerate(wsi_loader)):

            batch_probs   = self.softmax(self.model(batch_tiles.to(self.device)))
            batch_class   = torch.argmax(batch_probs, dim=1).cpu().detach().numpy().squeeze()
            batch_probs   = batch_probs.cpu().detach().numpy()

            del batch_tiles        
            torch.cuda.empty_cache()
            for i, tile_loc in enumerate(batch_tile_loc):
                try:
                    if batch_class[i] != 0:
                        loc_x_list.append(eval(tile_loc)[0])
                        loc_y_list.append(eval(tile_loc)[1])
                        preds_list.append(batch_class[i])
                        probs_list.append(max(batch_probs[i]))
                    if max(batch_probs[i]) > 0.9:
                        ret0[int(eval(tile_loc)[1]*self.target_size):int(eval(tile_loc)[1]+1)*self.target_size, int(eval(tile_loc)[0]*self.target_size):int(eval(tile_loc)[0]+1)*self.target_size] = np.ones((self.target_size,self.target_size))*batch_class[i]
                    continue   
                except:
                    if batch_class != 0:
                        loc_x_list.append(eval(tile_loc)[0])
                        loc_y_list.append(eval(tile_loc)[1])
                        preds_list.append(int(batch_class))
                        probs_list.append(max(batch_probs[i]))
                    if max(batch_probs[i]) > 0.9:
                        ret0[int(eval(tile_loc)[1]*self.target_size):int(eval(tile_loc)[1]+1)*self.target_size, int(eval(tile_loc)[0]*self.target_size):int(eval(tile_loc)[0]+1)*self.target_size] = np.ones((self.target_size,self.target_size))*int(batch_class)
                    continue
        
        count_dict = {'GIST':0, 'Leiomyoma':0, 'Schwannoma':0}
        for (pred, prob) in zip(preds_list, probs_list):
            if prob > 0.9:
                if pred == 1:
                    count_dict['GIST'] += 1
                elif pred == 2:
                    count_dict['Leiomyoma'] += 1
                elif pred == 3:
                    count_dict['Schwannoma'] += 1

        predict_class = max(count_dict, key=count_dict.get)
        # if the largest number is multipe ...
        if list(count_dict.values()).count(count_dict[predict_class]) >= 2:
            predict_class = 'Unclassfied'

        ret0 = cv2.resize(ret0, (slide_image.shape[1], slide_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        map_img_msk  = self.image_with_mask(slide_image.copy(), ret0)
        map_img_msk  = cv2.cvtColor(map_img_msk, cv2.COLOR_RGB2BGR)

        organ_name_list = [self.slide_organ]*len(loc_x_list)
        slide_name_list = [slide_name]*len(loc_x_list)
        df = pd.DataFrame(list(zip(organ_name_list, slide_name_list, loc_x_list, loc_y_list, preds_list, probs_list)), columns=['organ_name', 'slide_name', 'x_loc', 'y_loc', 'prediction', 'probability'])
        total_df = pd.concat([total_df, df])
        total_df.to_csv(csv_save_path, index=False)
        os.makedirs(f'{self.save_base_dir}/{self.slide_organ}', exist_ok=True)
        os.makedirs(f'{self.save_base_dir}/{self.slide_organ}/{predict_class}', exist_ok=True)
        cv2.imwrite(f'{self.save_base_dir}/{self.slide_organ}/{predict_class}/{slide_name}.png', map_img_msk)

        return total_df
