conda create -y -n shoebox python=3.7 pip
conda activate shoebox
conda install -y pytorch=1.6 torchvision cudatoolkit -c pytorch
conda install -y pandas scikit-image scikit-learn 
conda install -y -c conda-forge gdcm

pip install albumentations kaggle iterative-stratification omegaconf pretrainedmodels pydicom timm==0.1.30 transformers
#pip install git+https://github.com/JoHof/lungmask
