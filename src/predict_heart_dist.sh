export OMP_NUM_THREADS=2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love000.yaml predict_heart --dist --num-workers 2 --save-probas-dir ../data/train-heart-probas/ --annotations ../data/train/train_5fold.csv --checkpoint ../checkpoints/love000/fold0/MSNET2D_003_VM-0.9988.PTH
