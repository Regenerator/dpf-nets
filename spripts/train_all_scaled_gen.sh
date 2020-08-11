#!/bin/bash -l

python train_ae.py ./configs/autoencoding/all_scaled.yaml all_scaled_gen_model 400 0.000256
python train_ae.py ./configs/autoencoding/all_scaled.yaml all_scaled_gen_model 800 0.000064 --resume
python train_ae.py ./configs/autoencoding/all_scaled.yaml all_scaled_gen_model 1000 0.000016 --resume
python train_ae.py ./configs/autoencoding/all_scaled.yaml all_scaled_gen_model 1050 0.000004 --resume