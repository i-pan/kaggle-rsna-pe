export OMP_NUM_THREADS=2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mk3d/mk3d008.yaml train --dist --dist-val --num-workers 2 --kfold 0 --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mk3d/mk3d008.yaml train --dist --dist-val --num-workers 2 --kfold 1 --sync-bn --load-backbone ../checkpoints/mk013/fold1/NET2D_004_VM-0.0595.PTH --load-transformer ../checkpoints/seq103/fold1/TRANSFORMERCLS_009_VM-0.1530.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mk3d/mk3d008.yaml train --dist --dist-val --num-workers 2 --kfold 2 --sync-bn --load-backbone ../checkpoints/mk013/fold2/NET2D_004_VM-0.0627.PTH --load-transformer ../checkpoints/seq103/fold2/TRANSFORMERCLS_007_VM-0.1625.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mk3d/mk3d008.yaml train --dist --dist-val --num-workers 2 --kfold 3 --sync-bn --load-backbone ../checkpoints/mk013/fold3/NET2D_004_VM-0.0619.PTH --load-transformer ../checkpoints/seq103/fold3/TRANSFORMERCLS_007_VM-0.1586.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mk3d/mk3d008.yaml train --dist --dist-val --num-workers 2 --kfold 4 --sync-bn --load-backbone ../checkpoints/mk013/fold4/NET2D_004_VM-0.0652.PTH --load-transformer ../checkpoints/seq103/fold4/TRANSFORMERCLS_009_VM-0.1536.PTH

