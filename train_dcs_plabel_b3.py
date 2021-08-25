import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
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
label_path = os.path.join(data_path, 'train_labels.csv')
plabel_path = os.path.join(data_path, 'plabel.csv')

parser = argparse.ArgumentParser()

label_df = pd.read_csv(label_path)
plabel_df = pd.read_csv(plabel_path)
device = torch.device('cuda')

best_auc = .0

cfg = {
    'fold_num': 5,
    'seed': 2021,
    'model_arch': 'tf_efficientnet_b3',
    'img_size': 768,
    'epochs': 6,
    'train_batch_size': 80,
    'val_batch_size': 128,
    'T_0': 2,
    'T_mul': 1,
    'lr': 3e-4,
    'min_lr': 9e-5,
    'accum_iter': 1,
    'weight_decay': 1e-6,
    'num_workers': 16,
    'device': 'cuda',
    'device_num': 8,
    'skip_fold': 4
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
        A.Resize(cfg['img_size'], cfg['img_size']),

        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomBrightnessContrast(),
        ], p=0.5),

        # A.Cutout(num_holes=10, max_h_size=int(0.08*cfg['img_size']), max_w_size=int(0.08*cfg['img_size']), p=0.2),
        
        ToTensorV2(),
    ])

def valid_transforms():
    return A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        ToTensorV2(),
    ])

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
    train_df = pd.concat([train_df, plabel_df])
    train_df.reset_index(inplace=True, drop=True)
    val_df = label_df.loc[val_idx].reset_index(drop=True)

    train_dataset = SetiDataset(train_df, transform=train_transforms())
    val_dataset = SetiDataset(val_df, transform=valid_transforms())
    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False,
                            pin_memory=True, drop_last=False, num_workers=cfg['num_workers'])
    return train_loader, val_loader, train_df, val_df

def mixup(input, target):
    bs = input.size(0)
    
    lam_a = torch.ones(bs).cuda()
    target_b = target.clone()
    rand_index = torch.randperm(bs).cuda()
    target_b = target[rand_index]

    lam = np.random.beta(1, 1)
    lam_a = lam_a * lam
    lam_b = 1 - lam_a

    input = lam * input + (1-lam) * input[rand_index]

    return input.cuda(), target.cuda(), target_b.cuda(), lam_a.cuda(), lam_b.cuda()

def train(train_loader, net, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0.
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        images, target_a, target_b, lam_a, lam_b = mixup(images, labels)

        outputs = net(images)

        loss_a = criterion(outputs.view(-1), target_a)
        loss_b = criterion(outputs.view(-1), target_b)
        loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
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
    torch.save(state, os.path.join(project_path, f'checkpoint/seti-{model_arch}-fold{fold}-plabel-mixup-auc.pth'))

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
        save_model(net, fold, auc, test_loss)
    else:
        print(auc)

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
            net = torch.nn.DataParallel(net, device_ids=range(cfg['device_num']))
            checkpoint = torch.load(f"checkpoint/seti-{cfg['model_arch']}-fold{fold}-640-auc.pth")
            net.load_state_dict(checkpoint['net'])
            best_auc = checkpoint['auc']
        else:
            net = CustomModel(pretrained=True)
            net = torch.nn.DataParallel(net, device_ids=range(cfg['device_num']))

        net = net.cuda()

        optimizer = optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=cfg['T_0'], T_mult=cfg['T_mul'], eta_min=cfg['min_lr'], last_epoch=-1)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(cfg['epochs']):
            print(f'\nEpoch {epoch}')
            train(train_loader, net, optimizer, scheduler, criterion)
            validate(val_loader, net, criterion, fold, val_df.target.values)
        
        del net, train_loader, val_loader, optimizer, scheduler
        torch.cuda.empty_cache()

parser.add_argument('--resume', type=int, default=0)
args = parser.parse_args()

main_loop(args.resume)
