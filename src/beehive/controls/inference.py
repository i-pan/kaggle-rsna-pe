import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, os.path as osp
import gc
import re
import time

from collections import defaultdict
from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from torch.nn.parallel import DistributedDataParallel
from .. import builder
from ..utils import cudaify
from .datamaker import *
from tqdm import tqdm


def load_model(cfg):
    if cfg.model.params.pretrained:
        cfg.model.params.pretrained = False

    if cfg.model.params.load_backbone:
        cfg.model.params.load_backbone = None

    if cfg.model.params.load_transformer:
        cfg.model.params.load_transformer = None

    model = builder.build_model(cfg)
    weights = torch.load(cfg.checkpoint, map_location=lambda storage, loc: storage)
    weights = {re.sub(r'^module.', '', k) : v for k,v in weights.items()}
    if cfg.local_rank == 0:
        print(f'Loading checkpoint from {cfg.checkpoint} ...')
        model.load_state_dict(weights)
    if cfg.experiment.distributed:
        if cfg.experiment.cuda: 
            model.to(f'cuda:{cfg.local_rank}')
        model = DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)
    else:
        if cfg.experiment.cuda: 
            model.to(f'cuda:{cfg.local_rank}')
    model.eval()
    return model


class DummyProcess:

    def __init__(self, 
                 max_size,
                 crop_size,
                 window):
        self.max_size = max_size
        self.crop_size = crop_size
        self.window = window

    def apply_window(self, X):
        """For inference, we can skip the conversion to 8-bit
        and directly convert to [-1, 1] to save time. This is because
        we don't need to do data augmentation on 8-bit images. 
        """
        WL, WW = self.window
        upper, lower = WL+WW/2, WL-WW/2
        X.clip(lower, upper, out=X)
        X = X - np.min(X)
        if np.max(X) != 0: 
            X = X / np.max(X) 
        # X <> [0, 1]
        X = X - 0.5
        # X <> [-0.5, 0.5]
        X = X * 2.0
        # X <> [-1, 1]
        return X 

    def center_crop(self, X):
        # X.shape = (Z, C, H, W)
        h, w = self.crop_size 
        # Calculate center coordinate
        hc, wc = X.shape[-2] // 2, X.shape[-1] // 2
        h1, w1 = hc-h//2, wc-w//2
        h2, w2 = hc+h//2, wc+w//2
        return X[...,h1:h2,w1:w2]

    def process_image(self, X):
        # X.shape = (Z, H, W)
        # 1- Resize
        orig_max = np.max(X.shape[-2:])
        scale = self.max_size / orig_max
        if scale != 1:
            X = zoom(X, [1, scale, scale], order=1, prefilter=False)
        # 2- Crop
        X = self.center_crop(X)
        # 3- Window
        X = np.asarray([self.apply_window(_) for _ in X])
        return X


def convert_dataset_to_functions(dataset):
    """Use the PyTorch Dataset to create resize/preprocess 
    functions for faster preprocessing. 
    For processing the ENTIRE volume for 2D inference
    since this is one of the bottlenecks.
    """
    max_size = dataset.resize[0].max_size
    crop_x, crop_y = dataset.crop[0].height, dataset.crop[0].width
    return DummyProcess(max_size, 
                        (crop_x,crop_y), 
                        dataset.window)


class TrEnsemble(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.mean(torch.stack([m(x) for m in self.models], dim=0), dim=0)


def load_all_models(cfg):
    MODELS = cfg.models.keys()
    # 1- PE feature extractor
    # 2- PE transformer exam classifier 
    # 3- PE transformer slice classifier 
    # 4- Heart slice classifier
    # 5- RV/LV exam classifier
    # 6- RV/LV exam probability refiner
    # ...
    # cfg.models are paths to config files
    # cfg.checkpoints are paths to model checkpoints
    model_dict = defaultdict(list)
    for mtype in MODELS:
        configs = [OmegaConf.load(cfgpath) for cfgpath in getattr(cfg.models, mtype)]
        for i, each_cfg in enumerate(configs):
            each_cfg.experiment.cuda = cfg.experiment.cuda
            each_cfg.local_rank = cfg.local_rank
            each_cfg.checkpoint = getattr(cfg.checkpoints, mtype)[i]
            dset = get_dataset_for_test_prediction(each_cfg)
            if mtype in ['pe_feature_extractor', 'heart_slice_classifier']:
                dset = convert_dataset_to_functions(dset)
            model_dict[mtype] += [(load_model(each_cfg), dset)]
    num_pe_exam = len(model_dict['pe_exam_classifier'])
    num_pe_slice = len(model_dict['pe_slice_classifier'])
    num_pe_feat = len(model_dict['pe_feature_extractor'])
    if num_pe_exam > num_pe_feat:
        assert num_pe_exam % num_pe_feat == 0
        assert num_pe_slice % num_pe_feat == 0
        mx_exam = num_pe_exam // num_pe_feat
        mx_slice = num_pe_slice // num_pe_feat
        model_dict['pe_exam_classifier'] = [
            (
                TrEnsemble([j[0] for j in model_dict['pe_exam_classifier'][i:i+mx_exam]]),
                model_dict['pe_exam_classifier'][i][1]
            ) for i in range(0,num_pe_exam,mx_exam) 
        ]
        model_dict['pe_slice_classifier'] = [
            (
                TrEnsemble([j[0] for j in model_dict['pe_slice_classifier'][i:i+mx_slice]]),
                model_dict['pe_slice_classifier'][i][1]
            ) for i in range(0,num_pe_slice,mx_slice) 
        ]
    return model_dict


def extract_features(cfg, predict_proba=False):
    cfg.evaluate.batch_size = 1
    cfg.data.dataset.name = 'SimpleSeries'
    cfg.data.dataset.params.return_name = True
    loader = get_loader_for_feature_extraction(cfg, split_data=predict_proba)
    model = load_model(cfg)
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    for i, data in iterator:
        batch, tmp_input = data
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)
        # batch.shape = (1, Z, H, W), Z=# slices
        Z = batch.size(1)-2
        # Transform batch into pseudo-RGB images (3 continuous slices)
        batch = torch.cat([batch[:,i:i+3] for i in range(Z)])
        batch_indices = torch.split(torch.arange(batch.size(0)), 128)
        assert len(batch) == Z, f'`len(batch)` {len(batch)} must be equal to Z {Z}'
        with torch.no_grad():
            if predict_proba:
                output = torch.sigmoid(torch.cat([model(batch[b]) for b in batch_indices], dim=0))
            else:
                if cfg.experiment.distributed:
                    output = torch.cat([model.module.extract_features(batch[b]) for b in batch_indices], dim=0)
                else:
                    output = torch.cat([model.extract_features(batch[b]) for b in batch_indices], dim=0)
        if isinstance(tmp_input, list):
            tmp_input = tmp_input[0][0]
        series = tmp_input.split('/')[-2] + '.npy'
        study  = tmp_input.split('/')[-3]
        save_features_dir = osp.join(cfg.save_features_dir, study)
        try:
            if not osp.exists(save_features_dir): os.makedirs(save_features_dir)
        except Exception as e:
            print(e)
        np.save(osp.join(save_features_dir, series), output.cpu().numpy())


def extract_heart(cfg, predict_proba=False):
    cfg.evaluate.batch_size = 1
    cfg.data.dataset.name = 'SimpleSeries'
    cfg.data.add_padding = False
    cfg.data.dataset.params.return_name = True
    loader = get_loader_for_feature_extraction(cfg, split_data=predict_proba, heart_only=True)
    model = load_model(cfg)
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    for i, data in iterator:
        batch, tmp_input = data
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)
        # batch.shape = (1, Z, H, W), Z=# slices
        batch = batch.unsqueeze(1)
        batch = torch.cat([batch,batch,batch], dim=1)
        batch = F.interpolate(batch, size=(32,batch.size(-2),batch.size(-1)), mode='nearest')
        with torch.no_grad():
            if cfg.experiment.distributed:
                output = model.module.extract_features(batch)[0]
            else:
                output = model.extract_features(batch)[0]
        if isinstance(tmp_input, list):
            tmp_input = tmp_input[0][0]
        series = tmp_input.split('/')[-2] + '.npy'
        study  = tmp_input.split('/')[-3]
        save_features_dir = osp.join(cfg.save_features_dir, study)
        try:
            if not osp.exists(save_features_dir): os.makedirs(save_features_dir)
        except Exception as e:
            print(e)
        np.save(osp.join(save_features_dir, series), output.cpu().numpy())


def predict_pe_slice(cfg):
    extract_features(cfg, predict_proba=True)


def extract_features2(cfg):
    """extract_features above uses pseudo-RGB (2Dc) images (3 slices per image).
    This just uses 1 slice per image. 
    """
    bsize = 64
    cfg.evaluate.batch_size = 1
    cfg.data.dataset.name = 'SimpleSeries'
    cfg.data.dataset.params.add_padding = False
    cfg.data.dataset.params.return_name = True
    loader = get_loader_for_feature_extraction(cfg)
    model = load_model(cfg)
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    for i, data in iterator:
        batch, tmp_input = data
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)
        # batch.shape = (1, Z, H, W), Z=# slices
        batch = batch.unsqueeze(2)[0] #batch.shape = (Z, C, H, W)
        indices = torch.split(torch.arange(batch.size(0)), bsize)
        with torch.no_grad():
            if cfg.experiment.distributed:
                output = torch.cat([model.module.extract_features(b.unsqueeze(0)) for b in batch], dim=0)
            else:
                output = torch.cat([model.extract_features(b.unsqueeze(0)) for b in batch], dim=0)
        assert len(output) == len(batch)
        if isinstance(tmp_input, list):
            tmp_input = tmp_input[0][0]
        series = tmp_input.split('/')[-2] + '.npy'
        study  = tmp_input.split('/')[-3]
        save_features_dir = osp.join(cfg.save_features_dir, study)
        try:
            if not osp.exists(save_features_dir): os.makedirs(save_features_dir)
        except Exception as e:
            print(e)
        np.save(osp.join(save_features_dir, series), output.cpu().numpy())


def predict_features(cfg):
    cfg.evaluate.batch_size = 1
    cfg.data.dataset.name = 'FeatureDataset'
    cfg.data.dataset.params.return_name = True
    loader = get_loader_for_feature_prediction(cfg, 'pe_present_on_image')
    model = load_model(cfg)
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    series_ids, study_ids, probas, slice_indices = [],[],[],[]
    for i, data in iterator:
        batch, tmp_input = data
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)
        with torch.no_grad():
            output = torch.sigmoid(model(batch))
        tmp_input = tmp_input[0]
        series = tmp_input.split('/')[-1].replace('.npy', '')
        study  = tmp_input.split('/')[-2]
        series_ids += [series] * output.size(1)
        study_ids += [study] * output.size(1)
        probas += list(output.cpu().numpy()[0])
        slice_indices += list(range(output.size(1)))
    df = pd.DataFrame({
            'StudyInstanceUID': study_ids,
            'SeriesInstanceUID': series_ids,
            'SliceIndex': slice_indices,
            'pe_proba': probas
        })
    if not osp.exists(cfg.save_probas_dir): os.makedirs(cfg.save_probas_dir)
    df.to_csv(osp.join(cfg.save_probas_dir, f'pe_probas_{cfg.local_rank}.csv'), index=False)


def predict_features_cls(cfg):
    targets = [
        'negative_exam_for_pe',
        'indeterminate',
        'chronic_pe',
        'acute_and_chronic_pe',
        'central_pe',
        'leftsided_pe',
        'rightsided_pe'
    ] 
    cfg.evaluate.batch_size = 1
    cfg.data.dataset.name = 'FeatureDataset'
    cfg.data.dataset.params.return_name = True
    loader = get_loader_for_feature_prediction(cfg, targets)
    model = load_model(cfg)
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    series_ids, study_ids, probas = [], [], []
    for i, data in iterator:
        batch, tmp_input = data
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)
        with torch.no_grad():
            output = torch.sigmoid(model(batch))
        tmp_input = tmp_input[0]
        series = tmp_input.split('/')[-1].replace('.npy', '')
        study  = tmp_input.split('/')[-2]
        series_ids += [series]
        study_ids += [study]
        probas += list(output.cpu().numpy())
    probas = np.vstack(probas)
    df = pd.DataFrame(probas)
    df.columns = targets
    df['StudyInstanceUID'] = study_ids
    df['SeriesInstanceUID'] = series_ids
    if not osp.exists(cfg.save_probas_dir): os.makedirs(cfg.save_probas_dir)
    df.to_csv(osp.join(cfg.save_probas_dir, f'pe_cls_probas_{cfg.local_rank}.csv'), index=False)


def predict_heart(cfg):
    cfg.evaluate.batch_size = 1
    cfg.data.dataset.name = 'SimpleSeries'
    cfg.data.dataset.params.return_name = True
    loader = get_loader_for_feature_extraction(cfg)
    model = load_model(cfg)
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    series_ids, study_ids, probas, slice_indices = [],[],[],[]
    for i, data in iterator:
        batch, tmp_input = data
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)
        # batch.shape = (1, Z, H, W), Z=# slices
        Z = batch.size(1)-2
        # Transform batch into pseudo-RGB images (3 continuous slices)
        batch = [batch[:,i:i+3] for i in range(Z)]
        assert len(batch) == Z, f'`len(batch)` {len(batch)} must be equal to Z {Z}'
        with torch.no_grad():
            output = torch.cat([model(b) for b in batch], dim=0)
            output = torch.sigmoid(output)
        if isinstance(tmp_input, list):
            tmp_input = tmp_input[0][0]
        series = tmp_input.split('/')[-2]
        study  = tmp_input.split('/')[-3]
        series_ids += [series] * output.size(0)
        study_ids += [study] * output.size(0)
        probas += list(output.cpu().numpy()[:,1])
        slice_indices += list(range(output.size(0)))
    df = pd.DataFrame({
            'StudyInstanceUID': study_ids,
            'SeriesInstanceUID': series_ids,
            'SliceIndex': slice_indices,
            'heart_proba': probas
        })
    if not osp.exists(cfg.save_probas_dir): os.makedirs(cfg.save_probas_dir)
    df.to_csv(osp.join(cfg.save_probas_dir, f'heart_probas_{cfg.local_rank}.csv'), index=False)


def predict_rvlv(cfg):
    cfg.evaluate.batch_size = 1
    cfg.data.dataset.name = 'SeriesDataset'
    cfg.data.dataset.params.return_name = True
    loader = get_loader_for_series_prediction(cfg, 'rv_lv_ratio_gte_1')
    model = load_model(cfg)
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    series_ids, study_ids, probas = [], [], []
    for i, data in iterator:
        batch, tmp_input = data
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)
        with torch.no_grad():
            output = torch.sigmoid(model(batch))
        tmp_input = tmp_input[0][0]
        series = tmp_input.split('/')[-2].split('.')[0]
        study  = tmp_input.split('/')[-3]
        series_ids += [series]
        study_ids += [study]
        probas += list(output.cpu().numpy())
    probas = np.vstack(probas)
    df = pd.DataFrame(probas)
    df.columns = ['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']
    df['StudyInstanceUID'] = study_ids
    df['SeriesInstanceUID'] = series_ids
    if not osp.exists(cfg.save_probas_dir): os.makedirs(cfg.save_probas_dir)
    df.to_csv(osp.join(cfg.save_probas_dir, f'rvlv_probas_{cfg.local_rank}.csv'), index=False)


def get_loader_for_test_prediction(cfg, 
                                   fast_commit=False, 
                                   public_test=None,
                                   use_vtk=False):
    from ..data.datasets import SeriesPredict
    df = pd.read_csv(cfg.data.annotations)
    # df will just be a 3 columns for test set:
    #   StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID
    data_dir = cfg.data.data_dir
    df['folders'] = df.StudyInstanceUID+os.sep+df.SeriesInstanceUID 
    df['folders'] = [osp.join(data_dir, _) for _ in df.folders]
    if public_test:
        pt_df = pd.read_csv(public_test)
        pt_df = pt_df[np.asarray(['_' in i for i in pt_df['id']])]
        pt_df['StudyInstanceUID'] = pt_df['id'].map(lambda x: x.split('_')[0])
        df = df[~df.StudyInstanceUID.isin(pt_df.StudyInstanceUID)]
    if fast_commit:
        dataset = SeriesPredict(inputs=np.unique(df.folders.values)[:10], use_vtk=use_vtk)
    else:
        dataset = SeriesPredict(inputs=np.unique(df.folders.values), use_vtk=use_vtk)

    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'PREDICT : n={len(dataset)}')

    if len(dataset) == 0:
        return None

    loader = build_dataloader(cfg,
        dataset=dataset,
        mode='predict_full')

    return loader


def wide_to_long(df):
    ALL_LABELS = [
        'pe_present_on_image',
        'negative_exam_for_pe',
        'indeterminate',
        'chronic_pe',
        'acute_and_chronic_pe',
        'central_pe',
        'leftsided_pe',
        'rightsided_pe',
        'rv_lv_ratio_gte_1',
        'rv_lv_ratio_lt_1'
    ] 
    exam_labels = list(df[ALL_LABELS[1:]].values[0])
    new_df = pd.DataFrame({
        'id': [f'{df.StudyInstanceUID.values[0]}_{label}' for label in ALL_LABELS[1:]] + list(df.SOPInstanceUID),
        'label': exam_labels +list(df.pe_present_on_image),
    })
    if 'pos_frac' in df.columns:
        new_df['wt'] = WEIGHTS + list(df.pos_frac)
    return new_df


def predict_full(cfg, 
                 runtime=False, 
                 bsize=64, 
                 final_sub_only=True, 
                 fast_commit=False, 
                 public_test=None,
                 use_vtk=False,
                 debug=False):
    assert cfg.submission
    for k in cfg.models.keys():
        assert len(cfg.models[k]) == len(cfg.checkpoints[k])
    cfg.evaluate.batch_size = 1
    # We will be using 2 different windows
    # Thus set window to None for now
    loader = get_loader_for_test_prediction(cfg, fast_commit, public_test, use_vtk)
    if not loader:
        pd.read_csv(public_test).to_csv(cfg.submission, index=False)
        return

    # The preprocessor is built into the dataset
    # But we need to preprocess AFTER windowing
    # So we retrieve it from the dataset and set it to None
    models = load_all_models(cfg)
    for k in models.keys():
        print(f'{len(models[k])} {k} models found ...')
    # We want pseudo-RGB "stacked" images for the 2D image models
    models['pe_feature_extractor'][0][1].stacked = True
    models['heart_slice_classifier'][0][1].stacked = True
    exam_labels = [
        'negative_exam_for_pe',
        'indeterminate',
        'chronic_pe',
        'acute_and_chronic_pe',
        'central_pe',
        'leftsided_pe',
        'rightsided_pe',
        'rv_lv_ratio_gte_1',
        'rv_lv_ratio_lt_1'
    ] 
    iterator = tqdm(enumerate(loader), total=len(loader)) if cfg.local_rank == 0 else enumerate(loader)
    tic = time.time()

    dflist = []
    for i, data in iterator:
        if runtime:
            print(f'LOAD : {time.time()-tic:.2f}s')
        orig_array, tmp_input = data
        orig_array = orig_array[0].numpy()
        Z = len(orig_array)-2 # 1 slice of padding on top & bottom
        
        # 2D models use the same window [100, 700]
        # Process images ...
        tic = time.time()
        batch = models['pe_feature_extractor'][0][1].process_image(orig_array)
        # ==1- Extract PE features==
        # Transform batch into pseudo-RGB images (3 continuous slices)
        # We will use this for heart slice prediction later as well
        pseudo_rgb = np.asarray([batch[i:i+3] for i in range(Z)])
        assert len(pseudo_rgb) == Z, f'`len(pseudo_rgb)` {len(pseudo_rgb)} must be equal to Z {Z}'
        if runtime:
            print(f'1- PROCESS : {time.time()-tic:.2f}s')

        pseudo_rgb = torch.tensor(pseudo_rgb).float()
        if cfg.experiment.cuda:
            pseudo_rgb, _ = cudaify(pseudo_rgb, torch.tensor([0]), device=cfg.local_rank)

        # pseudo_rgb is now a torch tensor on CUDA 
        # We can use pseudo_rgb for heart_slice prediction as well ...
        # pseudo_rgb.shape = (Z, 3, H, W)

        tic = time.time()
        pe_features = []
        batch_indices = torch.split(torch.arange(pseudo_rgb.size(0)), bsize)
        for model_idx in range(len(models['pe_feature_extractor'])):
            with torch.no_grad():
                m = models['pe_feature_extractor'][model_idx][0]
                feat = torch.cat([m.extract_features(pseudo_rgb[bindex]) for bindex in batch_indices])
                feat = feat.unsqueeze(0)
                # feat.shape = (1, Z, dim_feat)
                pe_features += [feat]
        if runtime:
            print(f'1- PREDICT : {time.time()-tic:.2f}s')
        
        mask = torch.ones((1,pe_features[0].size(1))).to(pe_features[0].device)
        del batch

        # ==2- Get PE exam predictions==
        tic = time.time()
        pe_exam_preds = []
        for model_idx in range(len(models['pe_exam_classifier'])):
            with torch.no_grad():
                m = models['pe_exam_classifier'][model_idx][0]
                pe_exam_preds += [
                    torch.sigmoid(m((pe_features[model_idx],mask))).cpu().numpy()
                ]
        if runtime:
            print(f'2- PREDICT : {time.time()-tic:.2f}s')
        pe_exam_preds = np.mean(np.asarray(pe_exam_preds), axis=0)[0]

        # ==3- Get PE slice predictions==
        tic = time.time()
        pe_slice_preds = []
        for model_idx in range(len(models['pe_slice_classifier'])):
            with torch.no_grad():
                m = models['pe_slice_classifier'][model_idx][0]
                pe_slice_preds += [
                    torch.sigmoid(m((pe_features[model_idx],mask))).cpu().numpy()
                ]
        if runtime:
            print(f'3- PREDICT : {time.time()-tic:.2f}s')

        pe_slice_preds = np.mean(np.asarray(pe_slice_preds), axis=0)[0]

        # ==[OPTIONAL]- Use PE TDCNN for more PE exam predictions==
        if 'pe_exam_tdcnn' in models:
            # Get top 30% PE slices based on PE slice predictions
            tdcnn_dset = models['pe_exam_tdcnn'][0][1]
            tic = time.time()
            num_pe_slices = int(0.3*(len(orig_array)-2))
            mean_stack_proba = [np.mean(pe_slice_preds[i:i+num_pe_slices]) for i in range(0, len(pe_slice_preds)-num_pe_slices)]
            start_index = np.argmax(mean_stack_proba)
            end_index = start_index + num_pe_slices
            batch = pseudo_rgb[start_index:end_index]
            if len(batch) < tdcnn_dset.num_slices:
                padding = torch.zeros_like(batch[0]).unsqueeze(0)
                padding[...] = batch.min().item()
                padding = torch.cat([padding] * (tdcnn_dset.num_slices - len(batch)))
                batch = torch.cat([batch, padding])
            else:
                batch = F.interpolate(batch.permute(1,2,0,3), size=[tdcnn_dset.num_slices, batch.size(3)], mode='nearest')
                batch = batch.permute(2,0,1,3)
            crop_x, crop_y = tdcnn_dset.crop[0].height, tdcnn_dset.crop[0].width
            batch = F.interpolate(batch, size=[crop_x,crop_y], mode='bilinear', align_corners=True)
            if runtime:
                print(f'OPT- PROCESS : {time.time()-tic:.2f}s')

            tic = time.time()
            pe_tdcnn_preds = []
            for model_idx in range(len(models['pe_exam_tdcnn'])):
                with torch.no_grad():
                    m = models['pe_exam_tdcnn'][model_idx][0]
                    pe_tdcnn_preds += [
                        torch.sigmoid(m(batch.unsqueeze(0))).cpu().numpy()[0]
                    ]
            if runtime:
                print(f'OPT- PREDICT : {time.time()-tic:.2f}s')

            del batch
            pe_tdcnn_preds = np.mean(np.asarray(pe_tdcnn_preds), axis=0)
            #pe_exam_preds = (pe_exam_preds + pe_tdcnn_preds) / 2.0
            pe_exam_preds = np.average((pe_exam_preds, pe_tdcnn_preds),
                weights=np.asarray([[1,0.5],[0.5,1],[1,1],[1,1],[1,0.5],[1,0.5],[1,0.5]]).transpose(1,0),
                axis=0)

        # == 4- Get heart slice predictions==
        # Just use pseudo_rgb from PE feature extractor 
        # Resized in half
        
        pseudo_rgb = F.interpolate(pseudo_rgb, scale_factor=0.5, mode='bilinear', align_corners=True)
        tic = time.time()
        heart_slice_preds = []
        for model_idx in range(len(models['heart_slice_classifier'])):
            with torch.no_grad():
                m = models['heart_slice_classifier'][model_idx][0]
                heart_slice_preds += [
                    torch.sigmoid(torch.cat([m(pseudo_rgb[bindex]) for bindex in batch_indices])).cpu().numpy()
                ]

        if runtime:
            print(f'4- PREDICT : {time.time()-tic:.2f}s')

        heart_slice_preds = np.mean(np.asarray(heart_slice_preds), axis=0)[:,1]
        del pseudo_rgb

        #== 5- Get RV/LV predictions==
        # Take only heart slices
        tic = time.time()
        if np.sum(heart_slice_preds >= 0.5) == 0:
            # If heart slice predictions are all low, take top 25%
            num_slices = int(0.25*(len(orig_array)-2))
            mean_stack_proba = [np.mean(heart_slice_preds[i:i+num_slices]) for i in range(0, len(heart_slice_preds)-num_slices)]
            batch = orig_array[1:-1][np.argmax(mean_stack_proba):np.argmax(mean_stack_proba)+num_slices]
        else:
            batch = orig_array[1:-1][heart_slice_preds >= 0.5]
        batch = models['rvlv_exam_classifier'][0][1].process_image(batch)
        if runtime:
            print(f'5- PROCESS : {time.time()-tic:.2f}s')

        batch = torch.tensor(batch).float().unsqueeze(0)
        if cfg.experiment.cuda:
            batch, _ = cudaify(batch, torch.tensor([0]), device=cfg.local_rank)


        rvlv_features = []
        for model_idx in range(len(models['rvlv_exam_classifier'])):
            with torch.no_grad():
                m = models['rvlv_exam_classifier'][model_idx][0]
                rvlv_features += [m.extract_features(batch)]

        refined_rvlv = []
        pe_exam_feat = torch.from_numpy(pe_exam_preds).float().unsqueeze(0).to(rvlv_features[0].device) 
        for model_idx in range(len(models['rvlv_exam_linear'])):
            with torch.no_grad():
                m = models['rvlv_exam_linear'][model_idx][0]
                refined_rvlv += [
                    torch.sigmoid(m((rvlv_features[model_idx], 
                                     pe_exam_feat))).cpu().numpy()
                ]
        refined_rvlv = np.mean(np.asarray(refined_rvlv), axis=0)[0]

        # tic = time.time()
        # rvlv_preds =[]
        # for model_idx in range(len(models['rvlv_exam_classifier'])):
        #     with torch.no_grad():
        #         rvlv_preds += [torch.sigmoid(models['rvlv_exam_classifier'][model_idx][0](batch)).cpu().numpy()[0]]
        # rvlv_preds = np.mean(np.asarray(rvlv_preds), axis=0)
        # if runtime:
        #     print(f'5- PREDICT : {time.time()-tic:.2f}s')

        # # ==6- Refine RV/LV predictions==
        # exam_probas = np.concatenate((pe_exam_preds, rvlv_preds))
        # exam_probas = torch.tensor(exam_probas).float().unsqueeze(0)
        # refined_rvlv = []
        # if cfg.experiment.cuda:
        #     exam_probas, _ = cudaify(exam_probas, torch.tensor([0]), device=cfg.local_rank)
        # tic = time.time()
        # for model_idx in range(len(models['rvlv_exam_linear'])):
        #     with torch.no_grad():
        #         refined_rvlv += [torch.sigmoid(models['rvlv_exam_linear'][model_idx][0](exam_probas)).cpu().numpy()[0]]
        # if runtime:
        #     print(f'6- PREDICT : {time.time()-tic:.2f}s')
        # refined_rvlv = np.mean(np.asarray(refined_rvlv), axis=0)
        study_uid = tmp_input[0][0].split(os.sep)[-3]
        sries_uid = tmp_input[0][0].split(os.sep)[-2]
        slice_uid = [_[0].split(os.sep)[-1].split('.')[0] for _ in tmp_input]
        # Create wide DataFrame and transform to submission format later
        tmp_df = pd.DataFrame({
            'StudyInstanceUID': [study_uid] * len(slice_uid),
            'SeriesInstanceUID': [sries_uid] * len(slice_uid),
            'SOPInstanceUID': slice_uid
        })

        exam_probas = np.concatenate((pe_exam_preds, refined_rvlv))
        for label_idx, label in enumerate(exam_labels):
            tmp_df[label] = [exam_probas[label_idx]] * len(slice_uid)

        tmp_df['pe_present_on_image'] = pe_slice_preds
        tmp_df = enforce_label_consistency(tmp_df)
        tmp_df = wide_to_long(tmp_df)
        dflist += [tmp_df]
        # if i > 100:
        #     break
        del batch
        gc.collect()
        tic = time.time()

        if debug:
            if i > 200:
                break

    df = pd.concat(dflist)
    if not final_sub_only:
        df.to_csv(cfg.submission.replace('.csv','-wide-norules.csv'), index=False)
    #df = pd.concat([enforce_label_consistency(exam) for _, exam in df.groupby('StudyInstanceUID')])
    if not final_sub_only:
        df.to_csv(cfg.submission.replace('.csv','-wide.csv'), index=False)
    #df = pd.concat([wide_to_long(exam) for _, exam in df.groupby('StudyInstanceUID')])
    if public_test:
        pt_df = pd.read_csv(public_test)
        df = pd.concat([pt_df, df])
    test = pd.read_csv(cfg.data.annotations)
    errors = check_consistency(df, test)
    assert len(errors) == 0
    df.to_csv(cfg.submission, index=False)


def enforce_label_consistency(df):
    """Enforces label consistency requirements for predictions. 
    df : DataFrame consisting of a single exam's predictions
    """
    # 1- First, use negative_exam_for_pe to determine 
    # if the study should be positive or negative
    df = df.reset_index(drop=True)
    positive = df.negative_exam_for_pe.values[0] <= 0.5
    if positive: 
        # Ensure that at least 1 slice has PE>0.5
        if np.max(df.pe_present_on_image) <= 0.5:
            max_slice = np.argmax(df.pe_present_on_image.values)
            df['pe_present_on_image'].iloc[max_slice] = 0.50001
        # Ensure that one of rv_lv_ratio is >0.5
        if np.max(df[['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']].values) <= 0.5:
            gte1 = df.rv_lv_ratio_gte_1.values[0] > df.rv_lv_ratio_lt_1.values[0]
            if gte1:
                df['rv_lv_ratio_gte_1'] = 0.50001
            else:
                df['rv_lv_ratio_lt_1']  = 0.50001
        # Ensure that not BOTH of rv_lv_ratio >0.5
        if df.rv_lv_ratio_gte_1.values[0] > 0.5 and df.rv_lv_ratio_lt_1.values[0] > 0.5:
            df['rv_lv_ratio_gte_1'] = 0.4999
        # Ensure that one of central, rightsided, leftsided is >0.5
        location = ['central_pe', 'rightsided_pe', 'leftsided_pe']
        if np.max(df[location].values) <= 0.5:
            # Find the one with the highest value
            max_loc = np.argmax(df[location].values[0])
            df[location[max_loc]] = 0.50001
        # Ensure that both acute_and_chronic and chronic do NOT have >0.5
        if df.acute_and_chronic_pe.values[0] >= 0.5 and df.chronic_pe.values[0] >= 0.5:
            chronic = df.chronic_pe.values[0] > df.acute_and_chronic_pe.values[0]
            if chronic:
                df['acute_and_chronic_pe'] = 0.49999
            else:
                df['chronic_pe'] = 0.49999
        # We already know that negative_exam_for_pe is <= 0.5
        # Make sure that indeterminate is also <= 0.5
        if df.indeterminate.values[0] >= 0.5:
            df['indeterminate'] = 0.49999
    else:
        # Ensure that no slice has PE>0.5
        if df.pe_present_on_image.max() >= 0.5:
            df.loc[df.pe_present_on_image >= 0.5, 'pe_present_on_image'] = 0.49999
        # Ensure that indeterminate is not >= 0.5
        if df.indeterminate.values[0] >= 0.5:
            df['indeterminate'] = 0.49999
        positive_related_labels = ['central_pe', 'rightsided_pe', 'leftsided_pe',
            'acute_and_chronic_pe', 'chronic_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']
        for label in positive_related_labels:
            if df[label].values[0] >= 0.5:
                df[label] = 0.49999
    return df


def check_consistency(sub, test):
    
    '''
    Checks label consistency and returns the errors
    
    Args:
    sub   = submission dataframe (pandas)
    test  = test.csv dataframe (pandas)
    '''
    
    # EXAM LEVEL
    for i in test['StudyInstanceUID'].unique():
        df_tmp = sub.loc[sub.id.str.contains(i, regex = False)].reset_index(drop = True)
        df_tmp['StudyInstanceUID'] = df_tmp['id'].str.split('_').str[0]
        df_tmp['label_type']       = df_tmp['id'].str.split('_').str[1:].apply(lambda x: '_'.join(x))
        del df_tmp['id']
        if i == test['StudyInstanceUID'].unique()[0]:
            df = df_tmp.copy()
        else:
            df = pd.concat([df, df_tmp], axis = 0)
    df_exam = df.pivot(index = 'StudyInstanceUID', columns = 'label_type', values = 'label')
    
    # IMAGE LEVEL
    df_image = sub.loc[sub.id.isin(test.SOPInstanceUID)].reset_index(drop = True)
    df_image = df_image.merge(test, how = 'left', left_on = 'id', right_on = 'SOPInstanceUID')
    df_image.rename(columns = {"label": "pe_present_on_image"}, inplace = True)
    del df_image['id']
    
    # MERGER
    df = df_exam.merge(df_image, how = 'left', on = 'StudyInstanceUID')
    ids    = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    labels = [c for c in df.columns if c not in ids]
    df = df[ids + labels]
    
    # SPLIT NEGATIVE AND POSITIVE EXAMS
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())
    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]
    
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 
                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)
    rule1a['broken_rule'] = '1a'
    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 
                        (df_pos.rightsided_pe <= 0.5) & 
                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)
    rule1b['broken_rule'] = '1b'
    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 
                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)
    rule1c['broken_rule'] = '1c'
    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'

    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS
    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 
                         (df_neg.negative_exam_for_pe >  0.5)) | 
                        ((df_neg.indeterminate        <= 0.5)  & 
                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)
    rule2a['broken_rule'] = '2a'
    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 
                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |
                        (df_neg.central_pe           > 0.5) | 
                        (df_neg.rightsided_pe        > 0.5) | 
                        (df_neg.leftsided_pe         > 0.5) |
                        (df_neg.acute_and_chronic_pe > 0.5) | 
                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)
    rule2b['broken_rule'] = '2b'
    
    # MERGING INCONSISTENT PREDICTIONS
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)
    
    # OUTPUT
    print('Found', len(errors), 'inconsistent predictions')
    return errors






