#################
### get patch ###
#################

PATCH_SIZE      = 512
RESOLUTION      = 100
DOWN_LEVEL      = -1
ANNO_RATIO      = 0.05
TISSUE_RATIO    = 0.3
BATCH           = 20
SLIDE_LIST      = []  # ex) [./train_slide1.svs, ./train_slide2.ndpi]
SAVE_PATH       = ''  # ex) './wanted_patch_save_path/'  

########################
### get cutmix patch ###
########################    

NORMAL_LIST     = []  # ex) [./normal1.png, ./nomral2.png]
LEIOMYOMA_LIST  = []  # ex) [./leiomyoma1.png, ./leiomyoma2.png]
SCHWANNOMA_LIST = []  # ex) [./schwannoma1.png, ./schwannoma2.png]

NUM_NOR_SCH     = 20000
NUM_NOR_LEI     = 16000
NUM_LEI_SCH     = 16000
CUTMIX_PATH     = ''  # ex) './wanted_cutmixed_patch_save_path/'  