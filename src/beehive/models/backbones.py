import re
import timm
import torch
import torch.nn as nn


class Identity(nn.Module): 

    def forward(self, x): return x


##########
# MixNet #
##########
def mixnet_s(pretrained=True, **kwargs):
    model = timm.models.mixnet_s(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def mixnet_m(pretrained=True, **kwargs):
    model = timm.models.mixnet_m(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def mixnet_l(pretrained=True, **kwargs):
    model = timm.models.mixnet_l(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def mixnet_xl(pretrained=True, **kwargs):
    model = timm.models.mixnet_xl(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


###########
# ResNeSt #
###########
def resnest14(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest14d(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest26(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest26d(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest50(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest50d(pretrained=pretrained)
    if not kwargs.pop('use_maxpool', True):
        model.maxpool = Identity()
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest101(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest101e(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest200(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest200e(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest269(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest269e(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


##########
# ReXNet #
##########
def rexnet200(pretrained=True, **kwargs):
    model = timm.models.rexnet.rexnet_200(pretrained=pretrained)
    dim_feats = model.head.fc.in_features
    model.head.fc = Identity()
    return model, dim_feats


################
# EfficientNet #
################
def efficientnet_b0(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b0(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b1_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b1_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b1(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b1(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b2_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b2_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b2(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b2(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b3_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b3_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b3(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b3(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b4(pretrained=True, stride=[1,2,2,2,1,2,1], **kwargs):
    model = timm.models.tf_efficientnet_b4(pretrained=pretrained, **kwargs)
    assert len(stride) == len(model.blocks)
    for st, bl in zip(stride, model.blocks):
        if (st,st) != bl[0].conv_dw.stride:
            bl[0].conv_dw.stride = (st, st)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b5(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b5(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b6(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b6(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b6_ns(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b6_ns(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b7(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b7(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b8(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b8(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_l2(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_l2_ns(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


##########
# ResNet #
##########
def resnet34(pretrained=True, **kwargs):
    model = timm.models.resnet34(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


#######
# I3D #
#######
from .inception_v1_i3d import InceptionV1_I3D


def i3d(pretrained=True, **kwargs):
    model = InceptionV1_I3D()
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics400_se_rgb_inception_v1_seg1_f64s1_imagenet_deepmind-9b8e02b3.pth')
        weights = {k.replace('backbone.', '') : v for k,v in weights.items() if re.search('backbone', k)}
        model.load_state_dict(weights)
    dim_feats = 1024
    return model, dim_feats

from torchvision.models.video import mc3_18 as _mc3_18
from torchvision.models.video import r2plus1d_18 as _r2plus1d_18
from torchvision.models.video import r3d_18 as _r3d_18


def mc3_18(pretrained=True, **kwargs):
    model = _mc3_18(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def r2plus1d_18(pretrained=True, **kwargs):
    model = _r2plus1d_18(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def r3d_18(pretrained=True, **kwargs):
    model = _r3d_18(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


#######
# VMZ #
#######
from . import vmz


def ir_csn_152(pretrained=True, **kwargs):
    model = vmz.ir_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity() 
    return model, dim_feats


def ir_csn_101(pretrained=True, **kwargs):
    model = vmz.ir_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:23]
    return model, dim_feats


def ir_csn_50(pretrained=True, **kwargs):
    model = vmz.ir_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:6]
    return model, dim_feats


def ip_csn_152(pretrained=True, **kwargs):
    model = vmz.ip_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity() 
    return model, dim_feats


def ip_csn_101(pretrained=True, **kwargs):
    model = vmz.ip_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:23]
    return model, dim_feats


def ip_csn_50(pretrained=True, **kwargs):
    model = vmz.ip_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:6]
    return model, dim_feats


def r2plus1d_34(pretrained=True, **kwargs):
    model = vmz.r2plus1d_34(pretraining='32_ig65m' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity() 
    return model, dim_feats

