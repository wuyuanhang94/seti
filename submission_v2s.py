import os
from torch._C import device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
csv_path = os.path.join(data_path, 'sample_submission.csv')

test_df = pd.read_csv(csv_path)
device = torch.device('cuda')

cfg = {
    'model_arch': 'tf_efficientnetv2_s',
    'img_size': 640,
    'batch_size': 64*4,
    'device_num': 1
}

def get_test_file_path(image_id):
    return f'/datadisk/kg/seti/input/test/{image_id[0]}/{image_id}.npy'

class SetiDataset(Dataset):
    def __init__(self, df, transform=None):
        super(SetiDataset, self).__init__()
        self.df = df
        self.df['file_path'] = self.df.id.apply(get_test_file_path)
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
        self.fc = nn.Linear(n_features, 2)

    def forward(self, x):
        features = self.model(x)
        pooled_features = self.pooling(features).view(x.size(0), -1)
        return self.fc(pooled_features)

test_dataset = SetiDataset(test_df, test_transforms())
test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=12)

def subm(val_loader, net):
    net.eval()

    preds = []

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (inputs, labels) in progress_bar:
        x, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = net(x)
        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds[:, 1]

rets = []
models = glob.glob('checkpoint/seti-tf_efficientnetv2_s-fold[0-3]-640-auc.pth')
for model_name in models:
    net = CustomModel(pretrained=False)
    checkpoint = torch.load(model_name)['net']
    net.load_state_dict(checkpoint)
    net = net.cuda()

    preds = subm(test_loader, net)
    rets.append(preds)
    
pred = np.mean(np.array(rets), axis=0)
test_df.target = pred
test_df.drop(columns=['file_path'], inplace=True)
test_df.to_csv('subs/submission_v2s.csv', index=False)
