import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from . import backbones
from .arc import ArcMarginProduct
from .constants import POOLING
from .pooling import GeM, AdaptiveConcatPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from .sequence import *
from .wso import WSO2d, WSO3d

POOL2D_LAYERS = {
    'gem': GeM(p=3.0, dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': AdaptiveAvgPool2d(1),
    'max': AdaptiveMaxPool2d(1),
}


class Net2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 backbone_params={},
                 multisample_dropout=False,
                 feat_reduce=512,
                 pool='gem',
                 groups=1):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        if backbone != 'rexnet200':
            pool_layer = POOLING[backbone]
            pool_module = POOL2D_LAYERS[pool]
            if feat_reduce:
                pool_module = nn.Sequential(nn.Conv2d(dim_feats, feat_reduce, kernel_size=1, groups=groups), pool_module)
                dim_feats = feat_reduce
                self.feat_reduce = feat_reduce
            setattr(self.backbone, pool_layer, pool_module)
        self.ms_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1) 
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x

    def extract_features(self, x):
        return F.normalize(self.backbone(x).view(x.size(0), -1), dim=1)


class ArcNet(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 backbone_params={},
                 feat_reduce=512,
                 pool='gem',
                 groups=1):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        pool_layer = POOLING[backbone]
        pool_module = POOL2D_LAYERS[pool]
        if feat_reduce:
            pool_module = nn.Sequential(nn.Conv2d(dim_feats, feat_reduce, kernel_size=1, groups=groups), pool_module)
            dim_feats = feat_reduce
            self.feat_reduce = feat_reduce
        setattr(self.backbone, pool_layer, pool_module)
        self.dropout = nn.Dropout(p=dropout)
        self.arc = ArcMarginProduct(dim_feats, num_classes)

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return self.arc(features)

    def extract_features(self, x):
        return F.normalize(self.backbone(x), dim=1)


class MSNet2D(Net2D):

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)  
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features))
        return x 


class MSNet2D_V2(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 backbone_params={},
                 multisample_dropout=False,
                 feat_reduce=512,
                 pool='gem',
                 groups=1):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        pool_layer = POOLING[backbone]
        pool_module = POOL2D_LAYERS[pool]
        if feat_reduce:
            pool_module = nn.Sequential(nn.Conv2d(dim_feats, feat_reduce, kernel_size=1, groups=groups), pool_module)
            dim_feats = feat_reduce
            self.feat_reduce = feat_reduce
        setattr(self.backbone, pool_layer, pool_module)
        self.ms_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(dim_feats//3, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features).view(x.size(0), 3, features.size(1)//3)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features).view(x.size(0), 3, features.size(1)//3))
        return x.view(x.size(0), -1) 

    def extract_features(self, x):
        features = self.backbone(x)
        features = features.view(x.size(0), 3, -1)
        return F.normalize(features, dim=2)


class WSONet2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 backbone_params={},
                 multisample_dropout=False,
                 feat_reduce=512,
                 pool='gem',
                 groups=1,
                 wso_params={
                    'nch': 1,
                    'wl':  [100],
                    'ww':  [700],
                 }):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        pool_layer = POOLING[backbone]
        pool_module = POOL2D_LAYERS[pool]
        if feat_reduce:
            pool_module = nn.Sequential(nn.Conv2d(dim_feats, feat_reduce, kernel_size=1, groups=groups), pool_module)
            dim_feats = feat_reduce
            self.feat_reduce = feat_reduce
        setattr(self.backbone, pool_layer, pool_module)
        self.ms_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(dim_feats, num_classes)
        self.wso = WSO2d(**wso_params)

    def forward(self, x):
        features = self.extract_features(x, normalize=False)
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x

    def extract_features(self, x, normalize=True):
        x = self.wso(x)
        # After WSO, images should be [0, 255]
        x /= 255. 
        x -= 0.5
        x *= 2.0
        # Now, [-1, 1]
        features = self.backbone(x) 
        if normalize: features = F.normalize(features, dim=1)
        return features


class WSOMSNet2D(WSONet2D):

    def forward(self, x):
        if self.multislice_input:
            x = torch.cat([self.wso(x[:,i].unsqueeze(1)) for i in range(x.size(1))], dim=1)
        else:
            x = self.wso(x)
        # After WSO, images should be [0, 255]
        x /= 255. 
        x -= 0.5
        x *= 2.0
        features = self.backbone(x) 
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features))
        return x 


class Net3D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 backbone_params={},
                 multisample_dropout=False):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        self.ms_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)  
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x

    def extract_features(self, x):
        return F.normalize(self.backbone(x).view(x.size(0), -1), dim=1)


class TDCNN(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 load_transformer=None,
                 load_backbone=None,
                 feat_reduce=512,
                 pool='avg',
                 groups=1,
                 bn_eval=False,
                 backbone_params={},
                 transformer_params={}):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        pool_layer = POOLING[backbone]
        pool_module = POOL2D_LAYERS[pool]
        if feat_reduce:
            pool_module = nn.Sequential(nn.Conv2d(dim_feats, feat_reduce, kernel_size=1, groups=groups), pool_module)
            dim_feats = feat_reduce
            self.feat_reduce = feat_reduce
        setattr(self.backbone, pool_layer, pool_module)
        transformer_params['num_classes'] = num_classes
        self.transformer = TransformerCls(**transformer_params)

        if load_transformer:
            self.load_transformer(load_transformer)

        if load_backbone:
            self.load_backbone(load_backbone)

        self.bn_eval = bn_eval

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.bn_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def _load_weights(fp):
        weights = torch.load(fp, map_location=lambda storage, loc: storage)
        weights = {re.sub(r'^module.', '', k) : v for k,v in weights.items()}
        return weights

    def load_backbone(self, fp):
        print(f'Loading pretrained backbone weights from {fp} ...')
        weights = self._load_weights(fp)
        weights = {k : v for k,v in weights.items() if re.search(r'^backbone.', k)}
        weights = {re.sub(r'^backbone.', '', k) : v for k,v in weights.items()}
        self.backbone.load_state_dict(weights)

    def load_transformer(self, fp):
        print(f'Loading pretrained transformer weights from {fp} ...')
        weights = self._load_weights(fp)
        weights = {k : v for k,v in weights.items() if re.search(r'^transformer.', k)} 
        weights = {re.sub(r'^transformer.', '', k) : v for k,v in weights.items()}
        self.transformer.transformer.load_state_dict(weights)

    def forward(self, x):
        #features = torch.stack([self.backbone(x[:,:,i]) for i in range(Z)], dim=1)
        #features = self.combine(features.transpose(1,2)).squeeze(-1)
        B, Z, C, H, W = x.size()
        x = x.view(B*Z, C, H, W)
        features = self.backbone(x)
        features = features.view(B, Z, -1)
        mask = torch.from_numpy(np.ones((B,features.size(1)))).long().to(features.device)
        # features.shape = (B, Z, embedding_dim)
        output = self.transformer((features,mask))
        return output


class SimpleLinear(nn.Module):

    def __init__(self,
                 num_features, 
                 num_classes,
                 dropout,
                 multisample_dropout=False):
        super().__init__()
        self.linear  = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, num_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.ms_dropout = multisample_dropout

    def forward(self, x):
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(x)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(x))
        return x#return x[:,0] if self.linear.out_features == 1 else x


class LinBlock(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 layer_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features) if layer_norm else None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class HeartFeature(nn.Module):

    def __init__(self,
                 num_features=2048,
                 embedding_dims=(7,16),
                 hidden_dim=512,
                 num_classes=2,
                 n_layers=1,
                 dropout=0.2,
                 layer_norm=True):
        super().__init__()
        self.embed = nn.Linear(embedding_dims[0], embedding_dims[1])
        self.linear = LinBlock(num_features+embedding_dims[1], hidden_dim, layer_norm)
        hidden = []
        for i in range(n_layers):
            hidden += [LinBlock(hidden_dim, hidden_dim, layer_norm)]

        self.hidden = nn.Sequential(*hidden)
        self.drop = nn.Dropout(p=dropout)
        self.final = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x,p = x
        p = self.embed(p)
        x = torch.cat([x,p], dim=1)
        x = self.linear(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.final(x)
        return x   


class PoolCNN(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 seq_len=64,
                 backbone_params={},
                 multisample_dropout=False,
                 feat_reduce=512,
                 pool='gem',
                 groups=1):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        pool_layer = POOLING[backbone]
        pool_module = POOL2D_LAYERS[pool]
        if feat_reduce:
            pool_module = nn.Sequential(nn.Conv2d(dim_feats, feat_reduce, kernel_size=1, groups=groups), pool_module)
            dim_feats = feat_reduce
            self.feat_reduce = feat_reduce
        setattr(self.backbone, pool_layer, pool_module)
        self.ms_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.combine = nn.Linear(seq_len, 1)
        self.linear = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        #features = torch.stack([self.backbone(x[:,:,i]) for i in range(Z)], dim=1)
        #features = self.combine(features.transpose(1,2)).squeeze(-1)
        B, Z, C, H, W = x.size()
        x = x.view(B*Z, C, H, W)
        features = self.backbone(x)
        features = features.view(B, Z, -1)
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x


class PEProbRefiner(nn.Module):

    def __init__(self,
                 num_classes,
                 dropout=0.2,
                 dim=65):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p=dropout)
        self.lin2 = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.lin2(self.drop(self.lin1(x)))
        return x[:,0] if self.lin2.out_features == 1 else x



