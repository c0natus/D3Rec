import os, time
from datetime import timedelta
import argparse

import torch
from torch.utils.data import DataLoader

from dataset_load import load_data
from utils_train import *
from utils import *
from models import *


def main(args, dataset_dir_path, best_model_path):
    print(args)
    print(f'Use {args.device}')
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    dataset, _, _, test_dataset, matrix_F = load_data(args, dataset_dir_path)
    sp_train, sp_valid, sp_test = dataset.sp_train, dataset.sp_valid, dataset.sp_test
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    diffusion = Diffusion(
        steps=args.steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_scale=args.noise_scale,
        noise_schedule=args.noise_schedule,
        device=args.device)
    
    model = D3Rec(
        dims=args.dims,
        n_item=dataset.num_items,
        n_cate=dataset.num_cate,
        dim_step=args.dim_step,
        dropout=args.dropout).to(args.device)

    model.load_state_dict(torch.load(best_model_path, map_location='cpu').state_dict())

    for temperature in [1]:
        start = time.time()
        results = evaluate(args, model, diffusion, loader, sp_test, sp_train + sp_valid, args.topK, dataset.item_category, dataset.num_cate, temperature, is_best=True)
        print(f'Evaluation with w: {args.guide_w}, temperature {temperature} time: {str(timedelta(seconds=int(time.time() - start)))}')

        print_metric_results(args.topK, results)
                    

def get_args_parser():
    parser = argparse.ArgumentParser(description="D3Rec", add_help=True)

    ##### Training Setting #####
    parser.add_argument('--seed', default=1, type=int, help="Random seed")
    parser.add_argument('--cuda', default=0, type=int, help="GPU index")
    parser.add_argument('--batch_size', default=400, type=int, help="Batch size")
    parser.add_argument('--topK', default=[10, 20], type=int, nargs='+', help="Top k list")
    parser.add_argument('--save_model', action='store_true', help="Save model parameters.")

    ##### Data Setting #####
    parser.add_argument('--dataset_name', default='ml-1m', type=str, help="Dataset name")
    parser.add_argument('--valid_ratio', default=0.2, type=float, help="Valid ratio")
    parser.add_argument('--drop_num', default=20, type=int, help="Drop user whose history are less than drop_num")
    parser.add_argument('--test_w_valid', action='store_true',
                        help="True: test with train and valide data. False: test with train data.")

    ##### Model hyper parameter #####
    ### Auto Encoder
    parser.add_argument('--dims', type=int, default=[600, 200], nargs="+",
                        help="the dims for the classifier: n_item -> latent -> 1")
    parser.add_argument('--dropout', default=0.5, type=float, help="Drop interaction")
    parser.add_argument('--dim_step', default=10, type=int, help="Dimension of denoising step")
    parser.add_argument('--lamda', default=1, type=float, help="Peanlty of Orthogonal and Matching loss")
    parser.add_argument('--w_max', default=1, type=float, help="Range of reweight [w_min ~ w_max]")
    parser.add_argument('--w_min', default=0.2, type=float, help="Range of reweight [w_min ~ w_max]")

    ### Diffusion hyper parameter
    parser.add_argument('--beta_start', default=0.0001, type=float, help="Beta at time 0")
    parser.add_argument('--beta_end', default=0.02, type=float, help="Beta at time T")
    parser.add_argument('--noise_scale', default=0.1, type=float,
                        help="Strength of noise which is added into origin data.")
    parser.add_argument('--steps', default=100, type=int,
                        help="Denosing maximum step. (i.e., T value)")
    parser.add_argument('--snr', action='store_true', help="Use snr reweight or not")
    parser.add_argument('--sampling_steps', default=0, type=int, help="Denosing step at generate new data")
    parser.add_argument('--sampling_noise', action='store_true',
                        help="Apply uncertainty at generate new data. If False, mean is new data. Else, mean + variance is new data.")
    parser.add_argument('--noise_schedule', default="linear-var", type=str,
                        choices=['linear', 'linear-var', 'cosine', 'exp', 'binomial', 'sqrt'],
                        help="Beat noise schedule")

    ### Classifier-free
    parser.add_argument('--drop_div', default=0.1, type=float)
    parser.add_argument('--guide_w', default=0.5, type=float)

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    set_random_seed(random_seed=args.seed)
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    dataset_dir_path, best_model_path = get_paths(args)
    main(args, dataset_dir_path, best_model_path)
