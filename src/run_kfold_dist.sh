export OMP_NUM_THREADS=2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml train --dist --dist-val --num-workers 2 --kfold 0
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml train --dist --dist-val --num-workers 2 --kfold 1
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml train --dist --dist-val --num-workers 2 --kfold 2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml train --dist --dist-val --num-workers 2 --kfold 3
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml train --dist --dist-val --num-workers 2 --kfold 4

pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk020.yaml train --dist --dist-val --num-workers 2 --kfold 0
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk020.yaml train --dist --dist-val --num-workers 2 --kfold 1
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk020.yaml train --dist --dist-val --num-workers 2 --kfold 2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk020.yaml train --dist --dist-val --num-workers 2 --kfold 3
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk020.yaml train --dist --dist-val --num-workers 2 --kfold 4