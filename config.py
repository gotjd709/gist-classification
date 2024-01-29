###################
###### train ######
###################

CLASSES         = 4 
PATIENCE        = 5 
BATCH_SIZE      = 16
NUM_WORKER      = 4 
LOSS            = 'CrossEntropyLoss'
LR              = 1e-4 
OPTIMIZER       = 'Adam' 
EPOCH           = 100 

CUTMIX          = True
IMB_SAMPLER     = True

TRAIN_PATH      = ''  # ex) './train_dataset_pickle_path'
VALID_PATH      = ''  # ex) './valid_dataset_pickle_path'
TEST_PATH       = ''  # ex) './test_dataset_pickle_path' 
NOR_SCH_PATH    = ''  # ex) './normal_schwannoma_dataset_pickle_path' 
NOR_LEI_PATH    = ''  # ex) './normal_leiomyoma_dataset_pickle_path' 
LEI_SCH_PATH    = ''  # ex) './leiomyoma_schwannoma_dataset_pickle_path' 

### pickle file must be saved as follows [('./patch1.png', F.one_hot(torch.Tensor([n1]).to(torch.int64),4)), ('./patch2.png', F.one_hot(torch.Tensor([n2]).to(torch.int64),4))]

MODEL           = 'resent50'
MODEL_SAVE_PATH = ''  # ex) 

#########################
###### inference ######
#########################

TARGET_SLIDE    = []  # ex) [./test_slide1.svs, ./test_slide2.ndpi]
ORGAN_NAME      = ''  # ex) 'KUMC'
MODEL_PATH      = ''  # ex) './your_model_path.pth'
TARGET_MPP      = 1.0
TARGET_SIZE     = 512
OVERLAP         = 0
INFER_BATCH     = 32
SAVE_BASE_DIR   = ''  # ex) './results_save_path/'
CSV_SAVE_PATH   = ''  # ex) './results_csv_file_save_path.csv'