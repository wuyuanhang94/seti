import os

from albumentations.augmentations.transforms import Cutout, HorizontalFlip, RandomBrightnessContrast, VerticalFlip
from albumentations.core.composition import OneOf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader

import argparse
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

project_path = os.path.abspath('.')
data_path = os.path.join(project_path, 'input')
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')
csv_path = os.path.join(data_path, 'sample_submission.csv')

parser = argparse.ArgumentParser()

label_path = "/datadisk/kg/seti/train_df.csv"
label_df = pd.read_csv(label_path)
device = torch.device('cuda')

best_auc = .0

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model_arch': 'tf_efficientnet_b2',
    'img_size': 512,
    'epochs': 9,
    'train_batch_size': 64,
    'val_batch_size': 64,
    'T_0': 3,
    'T_mul': 1,
    'lr': 1e-5,
    'min_lr': 1e-6,
    'accum_iter': 1,
    'weight_decay': 1e-6,
    'num_workers': 12,
    'device': 'cuda',
    'device_num': 1,
    'skip_fold': 0
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_transforms():
    return A.Compose([
        # A.Resize(cfg['img_size'], cfg['img_size']),
        A.Resize(768, 768),
        A.RandomResizedCrop(cfg['img_size'], cfg['img_size'], scale=(0.9, 1.0)),
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        ], p=0.6),
            
        OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.3),

        OneOf([
            A.OpticalDistortion(distort_limit=0.8),
            A.GridDistortion(num_steps=5, distort_limit=0.8),
            A.ElasticTransform(alpha=3),
        ], p=0.3),
        
        A.Cutout(num_holes=5, max_h_size=int(0.08*cfg['img_size']), max_w_size=int(0.08*cfg['img_size']), p=0.4),
        ToTensorV2(),
    ])

def valid_transforms():
    return A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        ToTensorV2(),
    ])

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.02):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing 

    def forward(self, logits, labels):
        labels[labels == 1] = 1 - self.smoothing 
        labels[labels == 0] = self.smoothing 
        return F.binary_cross_entropy_with_logits(logits, labels)

class SetiDataset(Dataset):
    def __init__(self, df, transform=None):
        super(SetiDataset, self).__init__()
        self.df = df
        self.file_names = df.file_path.values
        self.labels = df.target.values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path)
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))

        image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx]).float()
        return image, label

class CustomModel(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(cfg['model_arch'], pretrained=pretrained, in_chans=1)

        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        features = self.model(x)
        pooled_features = self.pooling(features).view(x.size(0), -1)
        return self.fc(pooled_features)

def prepare_dataloader(fold):
    train_idx = label_df[label_df.fold != fold].index
    val_idx = label_df[label_df.fold == fold].index

    train_df = label_df.loc[train_idx].reset_index(drop=True)
    val_df = label_df.loc[val_idx].reset_index(drop=True)

    train_dataset = SetiDataset(train_df, transform=train_transforms())
    val_dataset = SetiDataset(val_df, transform=valid_transforms())
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True,
                              pin_memory=False, drop_last=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False,
                            pin_memory=False, drop_last=False, num_workers=cfg['num_workers'])
    return train_loader, val_loader, train_df, val_df

def train(train_loader, net, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0.
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        loss = criterion(outputs.view(-1), labels)
        loss /= cfg['accum_iter']
        loss.backward()
        if ((batch_idx+1) % cfg['accum_iter'] == 0 or batch_idx == len(train_loader)-1):
            optimizer.step()
            net.zero_grad()
            optimizer.zero_grad()

        train_loss += loss.item()
        description = 'Loss: %.3f' % (train_loss/(batch_idx+1))
        progress_bar.set_description(description)

    scheduler.step()

def save_model(net, fold, auc=.0, loss=.0):
    state = {
        'net': net.state_dict(),
        'auc': auc,
        'loss': loss
    }
    print(f'Saving auc: {auc}...')
    model_arch = cfg['model_arch']
    img_size = cfg['img_size']
    torch.save(state, os.path.join(project_path, f'checkpoint/seti-{model_arch}-fold{fold}-{img_size}-auc.pth'))

def validate(val_loader, net, criterion, fold, val_labels):
    global best_auc
    net.eval()
    test_loss = 0.
    preds = []

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    for batch_idx, (inputs, labels) in progress_bar:
        x, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = net(x)
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        loss = criterion(outputs.view(-1), labels)
        test_loss += loss.item()

        description = 'Loss: %.3f' % (test_loss/(batch_idx+1))
        progress_bar.set_description(description)

    preds = np.concatenate(preds)
    auc = roc_auc_score(val_labels, preds)

    test_loss /= len(val_loader)

    if auc > best_auc:
        best_auc = auc
        print(auc)
        save_model(net, fold, auc, test_loss)

def main_loop(resume):
    os.makedirs('checkpoint', exist_ok=True)
    seed_everything(cfg['seed'])
    torch.cuda.empty_cache()

    for fold in range(cfg['fold_num']):
        if fold < cfg['skip_fold']:
            continue
        print(f'{fold}th fold training starts')

        global best_auc
        best_auc = .0
        train_loader, val_loader, _, val_df = prepare_dataloader(fold)

        if resume:
            net = CustomModel(pretrained=False)
            checkpoint = torch.load(f"checkpoint/seti-{cfg['model_arch']}-fold{fold}-{cfg['img_size']}-auc.pth")
            net.load_state_dict(checkpoint['net'])
            best_auc = checkpoint['auc']
        else:
            net = CustomModel(pretrained=True)
        net = net.cuda()

        optimizer = optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=cfg['T_0'], T_mult=cfg['T_mul'], eta_min=cfg['min_lr'], last_epoch=-1)

        for epoch in range(cfg['epochs']):
            print(f'\nEpoch {epoch}')
            if epoch < 0:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = LabelSmoothing()

            train(train_loader, net, optimizer, scheduler, criterion)
            validate(val_loader, net, criterion, fold, val_df.target.values)
        
        del net, train_loader, val_loader, optimizer, scheduler
        torch.cuda.empty_cache()

parser.add_argument('--resume', type=int, default=1)
args = parser.parse_args()

main_loop(args.resume)
