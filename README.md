# RSNA STR Pulmonary Embolism Detection
2nd place solution for the RSNA STR Pulmonary Embolism Detection competition on Kaggle. 

Solution overview available at: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193401

## Environment
- 16 cores, 64 GB RAM
- 4 24 GB NVIDIA Quadro RTX 6000 GPU
- Python 3.7.7
- Anaconda
- PyTorch 1.6

## Setup Python environment
```
conda create -n rsna-pe python=3.7 pip
pip install -r requirements.txt
```

## Download data
```
mkdir data 
cd data
kaggle competitions download -c rsna-str-pulmonary-embolism-detection
unzip train.zip
```

## Extract-transform-load
```
cd src/etl
python 00_extract_metadata.py
python 01_create_cv_splits.py
```

## [Optional] Download trained checkpoints
```
mkdir checkpoints
cd checkpoints
kaggle datasets download -d https://www.kaggle.com/vaillant/rsna-str-pe-checkpoints
unzip rsna-str-pe-checkpoints.zip
```

Note: This uses distributed training across 4 GPUs. You may need to edit the commands in each script to match your environment. You will also likely have different checkpoint names if training models from scratch. Please change those as well for each script performing feature extraction/inference. 

## Train PE feature extractors
```
bash 0_run_kfold_dist.sh
```

## Extract PE features
```
bash 1_extract_kfold_dist.sh
```

## Train transformers
```
bash 2_run_transformer_kfold_dist.sh
```

## Train heart slice classifier
```
bash 3_run_heart_slices_dist.sh
```

## Obtain OOF predictions (PE/heart slice)
```
bash 4_predict_features_kfold_dist.sh
bash 5_predict_heart_slices_dist.sh
cd src/etl
python 08_create_train_df_with_pe_and_heart_probas.py
```

## Train time-dependent CNNs
```
bash 6_run_mk3d_dist.sh
```

## Train 3D RV/LV CNNs
```
bash 7_run_heart_dist.sh
```

## Extract heart features
```
bash 8_extract_rvlv.sh
```

## Obtain OOF predictions (PE/RV/LV exam)
```
bash 9_predict_features_cls_kfold_dist.sh
cd src/etl
python 09_create_proba_dataset.py
```

## Train linear model (refine RV/LV predictions based on PE exam labels)
```
bash 10_run_linear.py
```

## Inference
Please see public notebook at https://www.kaggle.com/vaillant/rsna-str-pe-submission.
