#!/bin/bash -l

python train_svr.py ./configs/svr/all.yaml all_svr_model 20 0.000256
python train_svr.py ./configs/svr/all.yaml all_svr_model 30 0.000064 --resume
python train_svr.py ./configs/svr/all.yaml all_svr_model 35 0.000016 --resume
python train_svr.py ./configs/svr/all.yaml all_svr_model 36 0.000004 --resume