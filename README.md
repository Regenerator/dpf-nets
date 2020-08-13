# Discrete Point Flow Networks 
Roman Klokov, Edmond Boyer, Jakob Verbeek

This repository contains the code for the **"Discrete Point Flow Networks for Efficient Point Cloud Generation"** [paper](https://arxiv.org/abs/2007.10170) accepted to 16th European Conference on Computer Vision, 2020.
It includes:
- preprocessing scripts for [ShapeNetCore55](https://www.shapenet.org/download/shapenetcore) and [ShapeNetAll13](http://3d-r2n2.stanford.edu/) datasets,
- implementation and training scripts for generative, autoencoding, and single-view reconstruction models presented in the paper.

## Environment
The code requires python-3.6 and these packages, (and was run using according versions):
- yaml-0.1.7
- numpy-1.17.2
- scipy-1.3.1
- pandas-0.25.3
- h5py-2.7.1
- opencv3-3.1.0
- pytorch-1.4.0
- torchvision-0.5.0

## Data preparation
Our point cloud sampler relies on data being stored in hdf5 format. So first of all the data should be converted to it.

### ShapeNetCore55
The data can be prepared for use with:
```
python preprocess_ShapeNetCore.py data_dir save_dir
```
Here `data_dir` should be the path to directory with unpacked [ShapeNetCore55.v2](https://www.shapenet.org/download/shapenetcore) dataset. The preprocessing script also relies on the official split file `all.csv` and on the data being organized in that directory as follows:
```
- data_dir
  - shapes
    - synsetId0
      - modelId0
        - models
          - model_normalized.obj
      - modelId1
      - ...
    - synsetId1
    - ...
  - all.csv
```
`save_dir` is the path to directory where repacked data is saved. There are mistakes in the official split file and the dataset such as missing shape directories and `.obj` files. Corresponding shapes are skipped during preprocessing and are not included in the repacked version of the dataset.

For reasons discussed in the paper we also randomly resplit the data into train/val/test sets with a separate script:
```
python resample_ShapeNetCore.py data_path
```
where `data_path` is the path to `.h5` output file of the previous script. It creates a separate `*_resampled.h5` file in the same directory.

### ShapeNetAll13
The images for this data are found [here](http://3d-r2n2.stanford.edu/). Instead of using voxel grids for these images we use original meshes from [ShapeNetCore55.v1](https://www.shapenet.org/download/shapenetcore). The data for SVR is prepared with:
```
python preprocess_ShapeNetAll.py shapenetcore.v1_data_dir shapenetall13_data_dir save_dir
```
where `shapenetcore.v1_data_dir` is structured as:
```
- shapenetcore.v1_data_dir
  - synsetId0
    - modelId0
      - model.obj
    - modelId1
    ...
  - synsetId1
  - ...
```
`shapenetall13_data_dir` is structured as:
```
- shapenetall13_data_dir
  - ShapeNetRendering
    - synsetId0
      - modelId0
        - rendering
          - 00.png
          - 01.png
          - ...
      - modelId1
      - ...
    - synsetId1
    - ...
```
and `save_dir` is the path to directory where repacked data is saved. The script first copies meshes from ShapeNetCore.v1 corresponding to images in ShapeNetAll13 and then repacks both images and meshes into two separate hdf5 files.

## Model usage
### Training
Each task and data setup have a separate config file, storing all the optional parameters of the model which are situated in `configs`. To use the model, you need to modify these configs by specifying `path2data` and `path2save` fields to directories which store repacked .h5 data files and which will store checkpoints accordingly. `path2save` use the following structure:
```
- path2save
  - models
    - DPFNets
```

For class-conditional generative models use:
```
./scripts/train_airplane_gen.sh
./scripts/train_car_gen.sh
./scripts/train_chair_gen.sh
```
For generative model trained with the whole dataset for autoencoding use:
```
./scripts/train_all_original_gen.sh
./scripts/train_all_scaled_gen.sh
```
For single-view reconstruction:
```
./scripts/train_all_svr.sh
```

### Evaluation
TBD

## Citation
```
@InProceedings{klokov20eccv,
  author    = {R. Klokov and E. Boyer and J. Verbeek},
  title     = {Discrete Point Flow Networks for Efficient Point Cloud Generation},
  booktitle = {Proceedings of the 16th European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```
