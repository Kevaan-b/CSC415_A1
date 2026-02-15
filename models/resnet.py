from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck
try:
    # torchvision<0.13
    from torchvision.models.resnet import model_urls
except ImportError:
    model_urls = None
try:
    # torchvision>=0.13
    from torchvision.models import (
        ResNet18_Weights,
        ResNet50_Weights,
        resnet18 as tv_resnet18,
        resnet50 as tv_resnet50,
    )
except ImportError:
    ResNet18_Weights = None
    ResNet50_Weights = None
    tv_resnet18 = None
    tv_resnet50 = None
from models.aux_models import aux_models

def _load_pretrained(model, arch):
    if model_urls is not None:
        model.load_state_dict(model_zoo.load_url(model_urls[arch]), strict=False)
        return

    if arch == "resnet18":
        builder = tv_resnet18
        weights = ResNet18_Weights
    elif arch == "resnet50":
        builder = tv_resnet50
        weights = ResNet50_Weights
    else:
        raise ValueError("Unsupported architecture: %s" % arch)

    if builder is None:
        raise RuntimeError("No torchvision pretrained loader is available")

    try:
        # torchvision>=0.13
        ref_model = builder(weights=weights.DEFAULT if weights is not None else None)
    except TypeError:
        # torchvision<0.13 fallback
        ref_model = builder(pretrained=True)
    model.load_state_dict(ref_model.state_dict(), strict=False)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
 
        self.feat_ch = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x), x
    
    def get_params(self, lr):
        return [{"params": self.parameters(), "lr": lr}]

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def resnet18(num_classes, num_domains, pretrained=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    if pretrained:
        _load_pretrained(model, "resnet18")
    
    dis_model, c_model, cp_model = aux_models(model.feat_ch, num_domains, num_classes, layers_dis=[1024, 256], layers_cls=[1024, 256])

    model.apply(freeze_bn)
    return model, dis_model, c_model, cp_model

def resnet50(num_classes, num_domains, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    if pretrained:
        _load_pretrained(model, "resnet50")

    dis_model, c_model, cp_model = aux_models(model.feat_ch, num_domains, num_classes, layers_dis=[1024, 256], layers_cls=[1024, 256])

    model.apply(freeze_bn)
    return model, dis_model, c_model, cp_model
