import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.models as models

GROUP_NORM_LOOKUP = {
    16: 1,  # -> channels per group: 8
    32: 1,  # -> channels per group: 8
    64: 1,  # -> channels per group: 8
    128: 1,  # -> channels per group: 16
    256: 1,  # -> channels per group: 16
    512: 1,  # -> channels per group: 16
    1024: 1,  # -> channels per group: 32
    2048: 1,  # -> channels per group: 64
}


class Net(nn.Module):
    def __init__(self, num_class=2, input_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, input_channels=2, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if ("resnet" in model_name) and ("3d" not in model_name):
        """ Resnet18
        """
        networks = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }
        model_ft = networks[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        model_ft.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 1), padding=(3, 3), bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif "3d" in model_name:
        from efficient_3DCNN.models.resnet import resnet18
        from efficient_3DCNN.models.mobilenetv2 import get_model
        model_ft = resnet18(in_channel=1, num_classes=num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 1), padding=(3, 3),
                                         bias=False)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet":
        """ Mobilenet
        """
        model_ft = models.mobilenet_v3_small(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.features[0][0] = nn.Conv2d(input_channels,
                                            model_ft.features[0][0].out_channels,
                                            kernel_size=model_ft.features[0][0].kernel_size,
                                            stride=(1, 1),
                                            padding=model_ft.features[0][0].padding,
                                            bias=False)
        model_ft.classifier[-1] = nn.Linear(model_ft.classifier[-1].in_features, num_classes)
        # num_ftrs = model_ft.classifier.in_features
        model_ft.classifier[-1] = nn.Linear(1024, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif 'efficientnet' in model_name:

        networks = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
        }
        model_ft = networks[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.features[0][0] = nn.Conv2d(input_channels,
                                            model_ft.features[0][0].out_channels,
                                            kernel_size=model_ft.features[0][0].kernel_size,
                                            stride=(1, 1),
                                            padding=model_ft.features[0][0].padding,
                                            bias=False)
        model_ft.classifier[-1] = nn.Linear(model_ft.classifier[-1].in_features, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    # model_ft = batch_norm_to_group_norm(model_ft)
    return model_ft, input_size


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer
