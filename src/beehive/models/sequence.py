import numpy as np
import torch
import torch.nn as nn

from transformers.modeling_distilbert import Transformer as _Transformer
from .pooling import GeM, AdaptiveConcatPool1d


class Config:

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)


class Transformer(nn.Module):

    def __init__(self,
                 num_classes,
                 embedding_dim=512,
                 hidden_dim=1024,
                 n_layers=4,
                 n_heads=16,
                 dropout=0.2,
                 attn_dropout=0.1,
                 seq_len=256,
                 output_attns=False,
                 act_fn='gelu',
                 chunk=False,
                 reverse=False):
        super().__init__()
        config = Config(**{
                'dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'dropout': dropout,
                'attention_dropout': attn_dropout,
                'output_attentions': output_attns,
                'activation': act_fn,
                'output_hidden_states': False,
                'chunk_size_feed_forward': 0
            })

        self.transformer = _Transformer(config)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.chunk = chunk
        self.reverse = reverse

    def classify(self, x):
        out = self.classifier(x)
        if self.classifier.out_features == 1:
            return out[...,0]
        else:
            return out

    def forward_tr(self, x, mask):
        output = self.transformer(x, attn_mask=mask, head_mask=[None]*x.size(1))
        return self.classify(output[0])

    def forward(self, x):
        x, mask = x
        if self.chunk and not self.training:
            chunk_size = 64
            # Create placeholder for predictions
            output = torch.zeros((x.size(0), x.size(1))).float().to(x.device)
            chunks = torch.zeros_like(output).to(x.device)
            indices = torch.arange(0, x.size(1), chunk_size//2)
            if x.size(1) - indices[-1] < 16:
                indices = indices[:-1]
            for ind, i in enumerate(indices):
                if ind == len(indices)-1:
                    tmpx, tmpmask = x[:,i:], mask[:,i:]
                else:
                    tmpx, tmpmask = x[:,i:i+chunk_size], mask[:,i:i+chunk_size]
                chunkout = self.forward_tr(tmpx, tmpmask)
                output[:,i:i+chunkout.size(1)] += chunkout
                chunks[:,i:i+chunkout.size(1)] += 1.0
            output /= chunks
            return output
        else:
            if not self.training and self.reverse:
                xrev = torch.flip(x, dims=(1,))
                mrev = torch.flip(mask, dims=(1,))
                reverseout = self.forward_tr(xrev, mrev)
                out = self.forward_tr(x, mask)
                return torch.mean(torch.stack([out, torch.flip(reverseout, dims=(1,))]), dim=0)
            return self.forward_tr(x, mask)




SEQ_POOLING = {
    'gem': GeM(dim=1),
    'concat': AdaptiveConcatPool1d(),
    'avg': nn.AdaptiveAvgPool1d(1),
    'max': nn.AdaptiveMaxPool1d(1)
}
class TransformerCls(nn.Module):

    def __init__(self,
                 num_classes,
                 embedding_dim=512,
                 hidden_dim=1024,
                 n_layers=4,
                 n_heads=16,
                 dropout=0.2,
                 attn_dropout=0.1,
                 output_attns=False,
                 act_fn='gelu',
                 pool=None):
        super().__init__()
        config = Config(**{
                'dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'dropout': dropout,
                'attention_dropout': attn_dropout,
                'output_attentions': output_attns,
                'activation': act_fn,
                'output_hidden_states': False,
                'chunk_size_feed_forward': 0
            })

        self.transformer = _Transformer(config)
        if pool in ('avg','concat','gem','max'):
            self.pool = SEQ_POOLING[pool]
            if pool == 'concat': embedding_dim *= 2
        else:
            self.pool = None

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x, mask = x
        output = self.transformer(x, attn_mask=mask, head_mask=[None]*x.size(1))
        if self.pool:
            output = self.pool(output[0].transpose(-1,-2))[:,:,0]
        else:
            output = output[0][:,0]
        output = self.classifier(output)
        return output[...,0] if self.classifier.out_features == 1 else output



