import torch
from torchvision.datasets import CIFAR10,CIFAR100
from torchvision import transforms
import numpy as np
from flamby.datasets.fed_isic2019 import FedIsic2019

def load_cifar10(config):
    data_path = config.data.root
    test_set = CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
        transforms.Resize((224,224))
    ]))
    return test_set

def load_cifar100(config):
    data_path = config.data.root
    test_set = CIFAR100(data_path, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761)),
        transforms.Resize((224,224))
    ]))
    return test_set

def load_fedisic():
    return FedIsic2019(train=False, pooled=True)