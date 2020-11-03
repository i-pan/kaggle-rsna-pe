import logging
import numpy as np
import omegaconf
import pandas as pd
import os, os.path as osp

from ..builder import build_dataset, build_dataloader


def get_train_val_test_splits(df, cfg): 
    i, o = cfg.data.inner_fold, cfg.data.outer_fold
    if isinstance(i, (int,float)):
        if cfg.local_rank == 0:
            logger = logging.getLogger('root')
            logger.info(f'<inner fold> : {i}')
            logger.info(f'<outer fold> : {o}')
        test_df = df[df.outer == o]
        df = df[df.outer != o]
        train_df = df[df[f'inner{o}'] != i]
        valid_df = df[df[f'inner{o}'] == i]
    else:
        if cfg.local_rank == 0:
            logger = logging.getLogger('root')
            logger.info('No inner fold specified ...')
            logger.info(f'<outer fold> : {o}')
        test_df = None
        train_df = df[df.outer != o]
        valid_df = df[df.outer == o]
    return train_df, valid_df, test_df


def prepend_filepath(lst, prefix): 

    return np.asarray([osp.join(prefix, item) for item in lst])


def stack_channels(df, label, stack_labels=False):
    """Stack 3 contiguous slices to form pseudo-RGB image
    """
    new_df_list = []
    for series, _df in df.groupby('SeriesInstanceUID'):
        _df = _df.sort_values('ImagePositionPatient_2').reset_index(drop=True)
        stacked = ['placeholder'] + list(_df.filepath) + ['placeholder']
        stacked = [stacked[i:i+3] for i in range(len(stacked)-2)]
        _df['filepath'] = stacked
        if stack_labels:
            stacked_labels = [0] + list(_df[label]) + [0]
            stacked_labels = [stacked_labels[i:i+3] for i in range(len(stacked_labels)-2)]
            _df[label] = stacked_labels
        new_df_list += [_df]
    return pd.concat(new_df_list)


def _stack_channels(df, label, stack_labels=False):
    """Stack 3 contiguous slices to form pseudo-RGB image
    """
    inputs, labels = [], []
    for series, _df in df.groupby('SeriesInstanceUID'):
        _df = _df.sort_values('ImagePositionPatient_2').reset_index(drop=True)
        stacked = ['placeholder'] + list(_df.filepath) + ['placeholder']
        stacked = [stacked[i:i+3] for i in range(len(stacked)-2)]
        inputs.extend(stacked)
        if stack_labels:
            stacked_labels = _df[label].values
            if isinstance(label, str):
                stacked_labels = np.asarray([0] + list(stacked_labels) + [0])
                stacked_labels = [stacked_labels[i:i+3] for i in range(len(stacked_labels)-2)]
            else:
                assert isinstance(label, (list,omegaconf.listconfig.ListConfig))
                padding = np.expand_dims(np.asarray([0]*len(label)), axis=0)
                stacked_labels = np.concatenate((padding,stacked_labels,padding))
                stacked_labels = [stacked_labels[i:i+3].flatten() for i in range(len(stacked_labels)-2)]
            labels.append(stacked_labels)
        else:
            labels.append(list(_df[label].values))
    #inputs = [j for i in inputs for j in i]
    if stack_labels:
        labels = [j for i in labels for j in i]
    return inputs, labels


def get_series_filepaths_and_labels(df, TARGETS):
    inputs, labels = [], []
    for series, _df in df.groupby('SeriesInstanceUID'):
        _df = _df.sort_values('ImagePositionPatient_2').reset_index(drop=True)
        inputs.append(list(_df.filepath))
        tmp_labels = _df[TARGETS].values[0]
        labels.append(tmp_labels)
    return inputs, labels


def get_pe_proba_sequences(df):
    inputs, labels = [], []
    for series, _df in df.groupby('SeriesInstanceUID'):
        _df = _df.sort_values('ImagePositionPatient_2').reset_index(drop=True)
        inputs.append(_df[['pe_proba']].values)
        labels.append(_df.pe_present_on_image.values)
    return inputs, labels


def get_top_pe_slices(df, top_pct):
    dflist = []
    for series, _df in df.groupby('SeriesInstanceUID'):
        _df = _df.sort_values('ImagePositionPatient_2').reset_index(drop=True)
        num_slices = int(top_pct*len(_df))
        mean_stack_proba = [np.mean(_df.pe_proba.values[i:i+num_slices]) for i in range(0, len(_df)-num_slices)]
        tmp_df = _df.iloc[np.argmax(mean_stack_proba):np.argmax(mean_stack_proba)+num_slices]
        assert len(tmp_df) == num_slices
        dflist += [tmp_df]
    return pd.concat(dflist)


def get_features_filepaths_and_labels(df, LABEL_COL, exam_label=False):
    inputs, labels, z_pos = [], [], []
    for series, _df in df.groupby('SeriesInstanceUID'):
        _df = _df.sort_values('ImagePositionPatient_2').reset_index(drop=True)
        inputs.append(osp.dirname(_df.iloc[0].filepath)+'.npy')
        if exam_label:
            labels.append(_df[LABEL_COL].values[0])
        else:
            labels.append(_df[LABEL_COL].values)
        z_pos.append(_df.ImagePositionPatient_2.values)
    return inputs, labels, z_pos


def get_train_val_dataloaders(cfg):
    INPUT_COL = 'filepath'

    if not cfg.data.targets or cfg.data.targets == 'all':
        LABEL_COL = [
            'pe_present_on_image', 
            'indeterminate',
            'chronic_pe',
            'acute_and_chronic_pe',
            'central_pe',
            'leftsided_pe',
            'rightsided_pe',
            'rv_lv_ratio_gte_1',
            'rv_lv_ratio_lt_1',
        ]
    else:
        LABEL_COL = cfg.data.targets

    df = pd.read_csv(cfg.data.annotations)
    if cfg.data.dataset.name not in ('ProbDataset', 'RVLVFeatures'):
        data_dir = cfg.data.data_dir
        df[INPUT_COL] = prepend_filepath(df[INPUT_COL], data_dir)

    if cfg.data.include_negatives_rvlv:
        # Use pseudolabels
        neg_df = df[df.negative_exam_for_pe == 1]
        neg_df['rv_lv_ratio_gte_1'] = neg_df.prob_rv_lv_ratio_gte_1
        neg_df['rv_lv_ratio_lt_1'] = neg_df.prob_rv_lv_ratio_lt_1

    if cfg.data.positives_only:
        if cfg.local_rank == 0:
            print('Using positive exams only ...')
        #df = df[df.negative_exam_for_pe == 0]
        pe_present = df[df.pe_present_on_image == 1].StudyInstanceUID.unique()
        df = df[df.StudyInstanceUID.isin(pe_present)]

    if cfg.data.heart_only:
        if cfg.local_rank == 0:
            print('Using heart slices only ...')
        df = df[df.heart_proba >= 0.5]

    if cfg.data.top_pe_slices: 
        if cfg.local_rank == 0:
            print(f'Using top {cfg.data.top_pe_slices*100}% predicted PE slices ...')
        df = get_top_pe_slices(df, top_pct=cfg.data.top_pe_slices)

    ######
    # 3D #
    ######
    if cfg.data.dataset.name == 'SeriesDataset':
        train_df, valid_df, _ = get_train_val_test_splits(df, cfg)
        if cfg.data.include_negatives_rvlv:
            train_df = pd.concat([train_df, neg_df])
        train_inputs, train_labels = get_series_filepaths_and_labels(train_df, LABEL_COL)
        valid_inputs, valid_labels = get_series_filepaths_and_labels(valid_df, LABEL_COL)
        train_stuff = dict(inputs=train_inputs, labels=train_labels)
        valid_stuff = dict(inputs=valid_inputs, labels=valid_labels)
    elif cfg.data.dataset.name == 'FeatureDataset':
        train_df, valid_df, _ = get_train_val_test_splits(df, cfg)
        train_inputs, train_labels, train_z_pos = get_features_filepaths_and_labels(train_df, LABEL_COL, exam_label=cfg.data.dataset.params.exam_level_label)
        valid_inputs, valid_labels, valid_z_pos = get_features_filepaths_and_labels(valid_df, LABEL_COL, exam_label=cfg.data.dataset.params.exam_level_label)
        train_stuff = dict(inputs=train_inputs, labels=train_labels)
        valid_stuff = dict(inputs=valid_inputs, labels=valid_labels)
        if cfg.data.use_z_position:
            train_stuff.update({'z_pos': train_z_pos})
            valid_stuff.update({'z_pos': valid_z_pos})
    elif cfg.data.dataset.name == 'ProbDataset':
        features = [
            'prob_negative_exam_for_pe',
            'prob_indeterminate',
            'prob_chronic_pe',
            'prob_acute_and_chronic_pe',
            'prob_central_pe',
            'prob_leftsided_pe',
            'prob_rightsided_pe',
            'prob_rv_lv_ratio_gte_1',
            'prob_rv_lv_ratio_lt_1'
        ]
        train_df, valid_df, _ = get_train_val_test_splits(df, cfg)
        train_inputs, train_labels = train_df[features].values, train_df[LABEL_COL].values
        valid_inputs, valid_labels = valid_df[features].values, valid_df[LABEL_COL].values       
        train_stuff = dict(inputs=train_inputs, labels=train_labels)
        valid_stuff = dict(inputs=valid_inputs, labels=valid_labels)
    elif cfg.data.dataset.name == 'RVLVFeatures':
        features = [
            'prob_negative_exam_for_pe',
            'prob_indeterminate',
            'prob_chronic_pe',
            'prob_acute_and_chronic_pe',
            'prob_central_pe',
            'prob_leftsided_pe',
            'prob_rightsided_pe'
        ]
        train_df, valid_df, _ = get_train_val_test_splits(df, cfg)
        train_inputs = (cfg.data.data_dir+'/'+train_df.StudyInstanceUID+'/'+train_df.SeriesInstanceUID+'.npy').values
        valid_inputs = (cfg.data.data_dir+'/'+valid_df.StudyInstanceUID+'/'+valid_df.SeriesInstanceUID+'.npy').values
        train_labels, valid_labels = train_df[LABEL_COL].values, valid_df[LABEL_COL].values
        train_probas, valid_probas = train_df[features].values, valid_df[features].values
        train_stuff = dict(inputs=train_inputs, labels=train_labels, probas=train_probas)
        valid_stuff = dict(inputs=valid_inputs, labels=valid_labels, probas=valid_probas)

    ######
    # 2D #
    ######
    elif cfg.data.dataset.name == 'PEProbDataset':
        assert cfg.data.positives_only
        train_df, valid_df, _ = get_train_val_test_splits(df, cfg)
        train_inputs = (cfg.data.data_dir+'/'+train_df.SeriesInstanceUID+'/'+train_df.SOPInstanceUID+'.npy').values
        valid_inputs = (cfg.data.data_dir+'/'+valid_df.SeriesInstanceUID+'/'+valid_df.SOPInstanceUID+'.npy').values
        train_labels, valid_labels = None, None
        train_stuff = dict(inputs=train_inputs, labels=train_labels)
        valid_stuff = dict(inputs=valid_inputs, labels=valid_labels)
    else:   
        # Creates pseudo-RGB images by stacking 3 contiguous slices channel-wise
        if cfg.data.stack:
            df = stack_channels(df, LABEL_COL, stack_labels=cfg.data.stack_labels)

        if cfg.data.dataset.params.return_weight:
            assert cfg.data.positives_only
            weights = df.groupby('SeriesInstanceUID').pe_present_on_image.mean()
            weights = weights.reset_index()
            weights.columns = ['SeriesInstanceUID', 'wt']
            df = df.merge(weights, on='SeriesInstanceUID')
            #df['wt'] *= 0.0736196319
            df['wt'] /= weights.wt.mean()

        other_pe_cols = [
            'chronic_pe',
            'acute_and_chronic_pe',
            'central_pe',
            'leftsided_pe',
            'rightsided_pe',
        ]

        df.loc[df.pe_present_on_image == 0, other_pe_cols] = 0
        train_df, valid_df, _ = get_train_val_test_splits(df, cfg)

        if cfg.data.upsample:
            pos_train = train_df[train_df.pe_present_on_image == 1]
            neg_train = train_df[train_df.pe_present_on_image == 0]
            upsample_frac = len(neg_train) / len(pos_train)
            if cfg.local_rank == 0:
                print(f'Upsampling positives by {upsample_frac:.2f} ...')
            pos_train = pos_train.sample(frac=upsample_frac, replace=True)
            train_df = pd.concat([pos_train, neg_train]).sample(frac=1, replace=False).reset_index(drop=True)

        del df

        # if cfg.data.stack:
        #     train_inputs, train_labels = stack_channels(train_df, LABEL_COL, stack_labels=cfg.data.stack_labels)
        #     valid_inputs, valid_labels = stack_channels(valid_df, LABEL_COL, stack_labels=cfg.data.stack_labels)
        # else:
        train_inputs = list(train_df[INPUT_COL])
        train_labels = train_df[LABEL_COL].values
        valid_inputs = list(valid_df[INPUT_COL])
        valid_labels = valid_df[LABEL_COL].values
        train_stuff = dict(inputs=train_inputs, labels=train_labels)
        valid_stuff = dict(inputs=valid_inputs, labels=valid_labels)
        if cfg.data.dataset.params.return_weight:
            train_stuff.update({'weight': train_df.wt.values})
            valid_stuff.update({'weight': valid_df.wt.values})

    train_dataset = build_dataset(cfg, data_info=train_stuff, mode='train')
    valid_dataset = build_dataset(cfg, data_info=valid_stuff, mode='valid')

    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'TRAIN : n={len(train_dataset)}')
        logger.info(f'VALID : n={len(valid_dataset)}')

    train_loader = build_dataloader(cfg,
        dataset=train_dataset,
        mode='train')
    valid_loader = build_dataloader(cfg,
        dataset=valid_dataset,
        mode='valid')

    return train_loader, valid_loader


def get_loader_for_feature_extraction(cfg, split_data=False, heart_only=False):
    INPUT_COL = 'filepath'
    df = pd.read_csv(cfg.data.annotations)
    if heart_only:
        df = df[df.heart_proba >= 0.5]
    if split_data:
        train_df, valid_df, test_df = get_train_val_test_splits(df, cfg)
        df = test_df.copy() if test_df else valid_df.copy()
        del train_df
        del valid_df
        del test_df 

    data_dir = cfg.data.data_dir
    df[INPUT_COL] = prepend_filepath(df[INPUT_COL], data_dir)
    assert cfg.data.dataset.name == 'SimpleSeries'
    
    inputs, _ = get_series_filepaths_and_labels(df, 'filepath')    
    labels = [0] * len(inputs)

    dataset = build_dataset(cfg, 
        data_info=dict(inputs=inputs, labels=labels),
        mode='extract')

    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'EXTRACT : n={len(dataset)}')

    loader = build_dataloader(cfg,
        dataset=dataset,
        mode='extract')

    return loader


def get_loader_for_feature_prediction(cfg, targets):
    INPUT_COL = 'filepath'
    df = pd.read_csv(cfg.data.annotations)
    train_df, valid_df, test_df = get_train_val_test_splits(df, cfg)
    df = test_df.copy() if test_df else valid_df.copy()
    del train_df
    del valid_df
    del test_df 
    data_dir = cfg.data.data_dir
    df[INPUT_COL] = prepend_filepath(df[INPUT_COL], data_dir)
    
    inputs, _, _ = get_features_filepaths_and_labels(df, targets)    
    labels = [0] * len(inputs)

    dataset = build_dataset(cfg, 
        data_info=dict(inputs=inputs, labels=labels),
        mode='predict')

    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'PREDICT : n={len(dataset)}')

    loader = build_dataloader(cfg,
        dataset=dataset,
        mode='predict')

    return loader


def get_loader_for_pe_slice_prediction(cfg, targets):
    INPUT_COL = 'filepath'
    df = pd.read_csv(cfg.data.annotations)
    train_df, valid_df, test_df = get_train_val_test_splits(df, cfg)
    df = test_df.copy() if test_df else valid_df.copy()
    del train_df
    del valid_df
    del test_df 
    data_dir = cfg.data.data_dir
    df[INPUT_COL] = prepend_filepath(df[INPUT_COL], data_dir)
    
    inputs, _ = get_series_filepaths_and_labels(df, targets)    
    labels = [0] * len(inputs)

    dataset = build_dataset(cfg, 
        data_info=dict(inputs=inputs, labels=labels),
        mode='predict')

    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'PREDICT : n={len(dataset)}')

    loader = build_dataloader(cfg,
        dataset=dataset,
        mode='predict')

    return loader


def get_loader_for_series_prediction(cfg, targets):
    INPUT_COL = 'filepath'
    df = pd.read_csv(cfg.data.annotations)
    train_df, valid_df, test_df = get_train_val_test_splits(df, cfg)
    df = test_df.copy() if test_df else valid_df.copy()
    del train_df
    del valid_df
    del test_df 
    data_dir = cfg.data.data_dir
    df[INPUT_COL] = prepend_filepath(df[INPUT_COL], data_dir)
    
    inputs, _ = get_series_filepaths_and_labels(df, targets)    
    labels = [0] * len(inputs)

    dataset = build_dataset(cfg, 
        data_info=dict(inputs=inputs, labels=labels),
        mode='predict')

    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'PREDICT : n={len(dataset)}')

    loader = build_dataloader(cfg,
        dataset=dataset,
        mode='predict')

    return loader


def get_dataset_for_test_prediction(cfg):
    """This lets me use the original training config files
    to construct datasets for the various models, from which
    I can use the resize/crop/preprocess attributes ...
    """
    dataset = build_dataset(cfg, 
        data_info=dict(inputs=[0], labels=[0]),
        mode='predict')
    dataset.augment = None

    return dataset
