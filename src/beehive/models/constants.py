POOLING = {
    'global_pool': [
        'resnest14',
        'resnest26',
        'resnest50',
        'resnest101',
        'resnest200',
        'resnest269',
        'rexnet200',
        'efficientnet_b0',
        'efficientnet_b1_pruned',
        'efficientnet_b1',
        'efficientnet_b2_pruned',
        'efficientnet_b2',
        'efficientnet_b3_pruned',
        'efficientnet_b3',
        'tf_efficientnet_b4',
        'tf_efficientnet_b5',
        'tf_efficientnet_b6_ns',
        'tf_efficientnet_b6',
        'tf_efficientnet_b7',
        'tf_efficientnet_b8',
        'tf_efficientnet_l2',
        'mixnet_s',
        'mixnet_m',
        'mixnet_l',
        'mixnet_xl',
        'resnet34'
    ]
}

POOLING = {vi : k for k,v in POOLING.items() for vi in v}