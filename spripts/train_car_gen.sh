#!/bin/bash -l

python train_ae.py ./configs/generation/car.yaml car_gen_model 1000 0.000256
python train_ae.py ./configs/generation/car.yaml car_gen_model 1500 0.000064 --resume
python train_ae.py ./configs/generation/car.yaml car_gen_model 1750 0.000016 --resume
python train_ae.py ./configs/generation/car.yaml car_gen_model 1800 0.000004 --resume