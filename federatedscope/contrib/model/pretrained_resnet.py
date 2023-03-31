from federatedscope.register import register_model
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn


class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        model = resnet18(weights='DEFAULT')
        layers = list(model.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.Linear(512, num_classes))
    
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

def ModelBuilder(model_config, local_data):

    model = MyNet(model_config.out_channels)
    return model

def call_resnet(model_config, local_data):
    if model_config.type == "resnet18":
        model = ModelBuilder(model_config, local_data)
        return model


register_model("resnet18", call_resnet)