import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

from torch._C import device
import pandas as pd
import numpy as np
import glob

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

project_path = os.path.abspath('.')
data_path = os.path.join(project_path, 'input')
csv_path = os.path.join(data_path, 'submission.csv')

test_df = pd.read_csv(csv_path)
device = torch.device('cuda')

cfg = {
    'model_arch': 'resnet18d',
    'img_size': 768,
    'batch_size': 128*2*2,
    'device_num': 8
}

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

def test_transforms():
    return A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        ToTensorV2(),
    ])

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

test_dataset = SetiDataset(test_df, test_transforms())
test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=16)

def subm(val_loader, net):
    net.eval()

    preds = []

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    for _, (inputs, labels) in progress_bar:
        x, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = net(x)
        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds

rets = []
for model_name in glob.glob(f'checkpoint/seti-{cfg["model_arch"]}-fold[0-5]-768-plabel-auc.pth'):
    net = CustomModel(pretrained=False)
    net = torch.nn.DataParallel(net, device_ids=range(cfg['device_num']))
    checkpoint = torch.load(model_name)['net']
    net.load_state_dict(checkpoint)
    net = net.cuda()

    preds = subm(test_loader, net)
    rets.append(preds)
    
pred = np.mean(np.array(rets), axis=0)

test_df.target = pred

test_df.drop(['file_path'], axis=1, inplace=True)
test_df.to_csv(f'subs/submission_{cfg["model_arch"]}_p_m.csv', index=False)
