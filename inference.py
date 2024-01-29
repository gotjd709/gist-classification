from util                   import InferenceWSI
from config                 import *
import torch.nn             as nn
import pandas               as pd

class ModelSetting(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pre_trained = model
        self.last_layer  = nn.Linear(1000, 1)
        
    def forward(self, x):
        x = self.pre_trained(x)
        x = self.last_layer(x)
        return x

if __name__ == '__main__':
    error_list = []
    total_df = pd.DataFrame(columns=['organ_name', 'slide_name', 'x_loc', 'y_loc', 'prediction', 'probability'])

    for slide_path in TARGET_SLIDE:
        clf = InferenceWSI(MODEL_PATH, TARGET_MPP, TARGET_SIZE, OVERLAP, ORGAN_NAME, SAVE_BASE_DIR, INFER_BATCH)
        total_df = clf.get_predict(slide_path, total_df, CSV_SAVE_PATH)