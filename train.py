from torchvision                                             import models
from functional                                              import name_path, Accuracy, Fscore, EarlyStopping, TrainEpoch, ValidEpoch, check_predictions
from datagen                                                 import dataloader_setting
from model.se_resnext                                        import se_resnext101
from config                                                  import *
import torch.optim                                           as optim
import torch.nn                                              as nn
import torch 
import os

class ModelSetting(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pre_trained = model
        self.last_layer  = nn.Linear(1000, CLASSES)
        
    def forward(self, x):
        x = self.pre_trained(x)
        x = self.last_layer(x)
        return x

def resnet50():
    model = models.resnet50(pretrained=True)
    model = ModelSetting(model)
    return model

def swin_transformer_tiny():
    model = models.swin_t(weights='IMAGENET1K_V1')
    model = ModelSetting(model)
    return model

def se_resnext101_32x4d():
    model = se_resnext101(pretrained='imagenet')
    model.last_linear = nn.Linear(in_features=2048*100, out_features=CLASSES, bias=True)
    return model

def swin_transformer_v2_base():
    model = models.swin_v2_b(weights='IMAGENET1K_V1')
    model = ModelSetting(model)
    return model

def model_setting(model_name):
    if model_name == 'resnet50':
        model = resnet50()
    elif model_name == 'swin_t':
        model = swin_transformer_tiny()
    elif model_name == 'se_resnext101':
        model = se_resnext101_32x4d()
    elif model_name == 'swin_b':
        model = swin_transformer_v2_base()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model


def train_dataset(model_name, cutmix, imb_sampler):
    # model setting
    model = model_setting(model_name)

    # dataloader setting
    train_loader, valid_loader, test_loader = dataloader_setting(cutmix, imb_sampler)

    # weight and log setting
    os.makedirs(MODEL_SAVE_PATH, exist_ok = True)

    # loss, metrics, optimizer and schduler setting
    loss = getattr(nn, LOSS)()
    optimizer = getattr(optim, OPTIMIZER)(params=model.parameters(), lr=LR)
    metrics = [Accuracy(CLASSES), Fscore(CLASSES,0), Fscore(CLASSES,1), Fscore(CLASSES,2), Fscore(CLASSES,3)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    patience = PATIENCE
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=MODEL_SAVE_PATH)

    train_epoch = TrainEpoch(
        model, 
        classes = CLASSES,
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
    )
    valid_epoch = ValidEpoch(
        model, 
        classes = CLASSES,
        loss=loss, 
        metrics=metrics, 
        verbose=True,
    )

    for i in range(0, EPOCH):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        early_stopping(valid_logs['loss'], model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    
    model = torch.load(MODEL_SAVE_PATH)

    test_epoch = ValidEpoch(
        model=model,
        classes = CLASSES,
        loss=loss,
        metrics=metrics,
        verbose=True,
    )
    test_epoch.run(test_loader)


if __name__ == '__main__':
    train_dataset(model_name=MODEL, cutmix=CUTMIX, imb_sampler=IMB_SAMPLER)