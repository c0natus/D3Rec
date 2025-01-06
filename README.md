# Controlling Diversity at Inference: Guiding Diffusion Recommender Models with Targeted Category Preferences

Gwangseok Han*, [Wonbin Kweon](https://wonbinkweon.github.io/)*, Minsoo Kim, Hwanjo Yu

Accepted to KDD 2025 Research Track!

Original implementation for [paper](https://arxiv.org/abs/2411.11240)

## Getting Started

Python 3.10

### Import conda environment

```bash
conda env create -f d3rec_env.yml
```

## Real-world experiments

### preprocess (It is already completed)
1. Move `./datasets/[dataset name]/clean_df_C20.pt` to `./Real-world/datasets/[dataset name]/`
2. Run `cd Real-world`
3. Run `python preprocessing.py --dataset_name [dataset name]`
3. Confirm 'Clean_C20_[6,2,2].pt' file.

### Training
1. Run `bash train.sh`
    - Tune hyper-parameters in train.sh
    - The best hyperparameter is noted in the comments.

### Inference
1. Run `bash inference.sh`
    - Set the hyper-parameters of the best model

## Semi-synthetic experiments

- Similar to the real-world experiments
