# D3Rec

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