pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob000.yaml train --dist --num-workers 2 --kfold 0
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob000.yaml train --dist --num-workers 2 --kfold 1
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob000.yaml train --dist --num-workers 2 --kfold 2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob000.yaml train --dist --num-workers 2 --kfold 3
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob000.yaml train --dist --num-workers 2 --kfold 4


pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob004.yaml train --dist --num-workers 2 --kfold 0
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob004.yaml train --dist --num-workers 2 --kfold 1
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob004.yaml train --dist --num-workers 2 --kfold 2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob004.yaml train --dist --num-workers 2 --kfold 3
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/prob/prob004.yaml train --dist --num-workers 2 --kfold 4
