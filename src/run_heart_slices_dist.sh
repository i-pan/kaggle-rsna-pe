export OMP_NUM_THREADS=2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love000.yaml train --dist --num-workers 2 --kfold 0
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love000.yaml train --dist --num-workers 2 --kfold 1
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love000.yaml train --dist --num-workers 2 --kfold 2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love000.yaml train --dist --num-workers 2 --kfold 3
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love000.yaml train --dist --num-workers 2 --kfold 4


pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love001.yaml train --dist --num-workers 2 --kfold 0
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love001.yaml train --dist --num-workers 2 --kfold 1
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love001.yaml train --dist --num-workers 2 --kfold 2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love001.yaml train --dist --num-workers 2 --kfold 3
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love001.yaml train --dist --num-workers 2 --kfold 4

pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/love/love003.yaml train --dist --num-workers 2 --kfold 0
