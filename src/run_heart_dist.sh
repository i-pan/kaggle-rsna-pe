export OMP_NUM_THREADS=2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart018.yaml train --dist --num-workers 2 --kfold 0 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart018.yaml train --dist --num-workers 2 --kfold 1 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart018.yaml train --dist --num-workers 2 --kfold 2 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart018.yaml train --dist --num-workers 2 --kfold 3 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart018.yaml train --dist --num-workers 2 --kfold 4 --dist-val --sync-bn

pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart021.yaml train --dist --num-workers 2 --kfold 0 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart021.yaml train --dist --num-workers 2 --kfold 1 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart021.yaml train --dist --num-workers 2 --kfold 2 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart021.yaml train --dist --num-workers 2 --kfold 3 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart021.yaml train --dist --num-workers 2 --kfold 4 --dist-val --sync-bn

pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart023.yaml train --dist --num-workers 2 --kfold 0 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart023.yaml train --dist --num-workers 2 --kfold 1 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart023.yaml train --dist --num-workers 2 --kfold 2 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart023.yaml train --dist --num-workers 2 --kfold 3 --dist-val --sync-bn
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/heart/heart023.yaml train --dist --num-workers 2 --kfold 4 --dist-val --sync-bn
