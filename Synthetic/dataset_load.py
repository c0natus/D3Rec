import os
import torch
import numpy as np
from preprocessing import PreProcess
from torch.utils.data import Dataset


class D3RecData(Dataset):
    def __init__(self, sp_data, prob_in, prob_pred):
        self.sp_data = sp_data
        self.prob_in = prob_in
        self.prob_pred = prob_pred

    def __getitem__(self, idx):
        item = torch.FloatTensor(self.sp_data.getrow(idx).toarray()[0])
        prob_in = torch.Tensor(self.prob_in.iloc[idx]['user_pref']).to(torch.float32)
        prob_pred = torch.Tensor(self.prob_pred.iloc[idx]['user_pref']).to(torch.float32)
        return item, prob_in, prob_pred

    def __len__(self):
        return self.sp_data.shape[0]


def load_data(args, dir_path):
    dataset = PreProcess(args, dir_path, use_cache=True)

    train_dataset = D3RecData(dataset.sp_train, dataset.df_user_pref_train, dataset.df_user_pref_valid)
    valid_dataset = D3RecData(dataset.sp_train, dataset.df_user_pref_train, dataset.df_user_pref_valid)
    if args.test_w_valid:
        test_dataset = D3RecData(dataset.sp_train + dataset.sp_valid, dataset.df_user_pref_train_valid,
                               dataset.df_user_pref_test)
    else:
        test_dataset = D3RecData(dataset.sp_train, dataset.df_user_pref_train, dataset.df_user_pref_test)

    # shape: (category, num_items)
    matrix_F = torch.tensor(np.array(dataset.matrix_F), dtype=torch.float32, device=args.device)

    return dataset, train_dataset, valid_dataset, test_dataset, matrix_F


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Data loader", add_help=True)
    parser.add_argument('--cuda', default=0, type=int, help="GPU index")
    parser.add_argument('--dataset_name', default='ml-1m', type=str, help="Dataset name")
    parser.add_argument('--test_w_valid', action='store_true',
                        help="True: test with train and valide data. False: test with train data.")
    parser.add_argument('--drop_num', default=20, type=int, help="Drop user whose history are less than drop_num")
    parser.add_argument('--valid_ratio', default=0.2, type=float, help="Valid ratio")

    args = parser.parse_args()

    dir_path = os.path.join(os.getcwd(), 'dataset', args.dataset_name)
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print(f'Use: {args.device}')
    dataset, train_dataset, valid_dataset, test_dataset, matrix_F = load_data(args, dir_path)