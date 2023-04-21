import os
import sys
sys.path.append('/nvme/lisiyi/project/new_exp/FederatedScope')

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np

from federatedscope.contrib.model import pretrained_resnet as ResNetLib
from federatedscope.contrib.model import distill_resnet as d_ResNetLib
from federatedscope.core.auxiliaries.ReIterator import ReIterator

#### Load Model
deviceID='cuda:0'
method = 'fedavg'
dataset_name = 'CIFAR10@torchvision'
out_channels = 10

raw_method_lib = ['fedavg','fedem']
param_file_dir = '/nvme/lisiyi/project/final_model/'
param_file_path = param_file_dir+method+'.pt'
model = ResNetLib.MyNet(out_channels, False) if method in raw_method_lib else d_ResNetLib.MyNet(out_channels, False)
assert os.path.exists(param_file_path)
ckpt = torch.load(param_file_path, map_location=deviceID)
model.to(device=deviceID)
model.load_state_dict(ckpt['model'])
model.eval()

#### Load Test Set
data_path = '/nvme/lisiyi/project/data/'
test_set = CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
    transforms.Resize((224,224))
]))
batchsize=256
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, shuffle=True)
itr_round = len(test_set)//batchsize + int(len(test_set)%batchsize>0)

####starting eval
ys_true = []
ys_prob = []
for _ in range(itr_round) :
    data_batch = next(ReIterator(test_loader))
    x, label = [xy.to(deviceID) for xy in data_batch]
    if method in raw_method_lib:
        pred = model(x)
    else :
        pred, feature = model(x)
    if len(label.size()) == 0:
        label = label.unsqueeze(0)
    ys_true.append(label.detach().cpu().numpy())
    ys_prob.append(pred.detach().cpu().numpy())
ys_prob=np.concatenate(ys_prob)
ys_true=np.concatenate(ys_true)
ys_pred=np.argmax(ys_prob, axis=1)

is_labeled = ys_true[:]==ys_true[:]
correct = ys_true[is_labeled]==ys_pred[is_labeled]
acc_rate = float(np.sum(correct))/len(correct)
print(acc_rate)