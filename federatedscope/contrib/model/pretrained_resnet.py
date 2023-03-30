from federatedscope.register import register_model
from torchvision.models import resnet18, ResNet18_Weights



def ModelBuilder(model_config, local_data):

    model = resnet18(weights=ResNet18_Weights.DEFAULT, num_classes=model_config.out_channels)

    return model

def call_resnet(model_config, local_data):
    if model_config.type == "resnet":
        model = ModelBuilder(model_config, local_data)
        return model


register_model("resnet", call_resnet)