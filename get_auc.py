import torch
import glob

for fn in glob.glob("checkpoint/*resnext*.pth"):
    print(fn, torch.load(fn)['auc'])
