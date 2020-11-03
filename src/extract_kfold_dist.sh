export OMP_NUM_THREADS=2
# pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract --dist --num-workers 2 --save-features-dir ../data/train-features/mk012/fold0 --checkpoint ../checkpoints/mk012/fold0/MSNET2D_004_VM-0.0844.PTH
# pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract --dist --num-workers 2 --save-features-dir ../data/train-features/mk012/fold1 --checkpoint ../checkpoints/mk012/fold1/MSNET2D_004_VM-0.0808.PTH
# pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract --dist --num-workers 2 --save-features-dir ../data/train-features/mk012/fold2 --checkpoint ../checkpoints/mk012/fold2/MSNET2D_004_VM-0.0863.PTH
# pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract --dist --num-workers 2 --save-features-dir ../data/train-features/mk012/fold3 --checkpoint ../checkpoints/mk012/fold3/MSNET2D_004_VM-0.0880.PTH
# pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract --dist --num-workers 2 --save-features-dir ../data/train-features/mk012/fold4 --checkpoint ../checkpoints/mk012/fold4/MSNET2D_004_VM-0.0887.PTH

pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract_features --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk013/fold0 --checkpoint  ../checkpoints/mk013/fold0/NET2D_004_VM-0.0611.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract_features --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk013/fold1 --checkpoint  ../checkpoints/mk013/fold1/NET2D_004_VM-0.0595.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract_features --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk013/fold2 --checkpoint  ../checkpoints/mk013/fold2/NET2D_004_VM-0.0627.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract_features --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk013/fold3 --checkpoint  ../checkpoints/mk013/fold3/NET2D_004_VM-0.0619.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk013.yaml extract_features --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk013/fold4 --checkpoint  ../checkpoints/mk013/fold4/NET2D_004_VM-0.0652.PTH

pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk014.yaml extract_features --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk014/fold0 --checkpoint  ../checkpoints/mk014/fold0/NET2D_004_VM-0.0682.PTH


pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk015.yaml extract_features2 --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk015-2/fold0 --checkpoint ../checkpoints/mk015/fold0/WSONET2D_004_VM-0.0620.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk015.yaml extract_features2 --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk015/fold1 --checkpoint ../checkpoints/mk015/fold1/WSONET2D_004_VM-0.0614.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk015.yaml extract_features2 --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk015/fold2 --checkpoint ../checkpoints/mk015/fold2/WSONET2D_004_VM-0.0640.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk015.yaml extract_features2 --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk015/fold3 --checkpoint ../checkpoints/mk015/fold3/WSONET2D_004_VM-0.0654.PTH
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk015.yaml extract_features2 --dist --dist-val --num-workers 2 --save-features-dir ../data/train-features/mk015/fold4 --checkpoint ../checkpoints/mk015/fold4/WSONET2D_004_VM-0.0641.PTH