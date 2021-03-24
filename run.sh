# python train.py -d ddi -e ddi1_3hop  --gpu=0 --hop=3 --batch=256 --emb_dim=32 -b=10

# python train.py -d ddi -e ddi1_2hop_custom  --gpu=1 --hop=2 --batch=256 --emb_dim=32 -b=10

# python train.py -d ddi -e ddi1_2hop_16  --gpu=0 --hop=2 --batch=256 --emb_dim=16 -b=10

# python train.py -d ddi -e ddi1_2hop_8  --gpu=0 --hop=2 --batch=256 --emb_dim=8 -b=10

# python train.py -d ddi -e ddi1_2hop_64  --gpu=0 --hop=2 --batch=256 --emb_dim=64 -b=10

# python train.py -d ddi -e ddi_2hop_custom_nohetero --gpu=2 --hop=2 --batch=256 --emb_dim=32 -b=10

python train.py -d ddi -e ddi_hop3 --gpu=0 --hop=3 --batch=256 --emb_dim=32 -b=10 --load_model