import os
import ast
import random
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class PreProcess():
    def __init__(self, args, dir_path, use_cache=False):
        self.dataset_name = args.dataset_name
        clean_dataset_split_path = os.path.join(dir_path,
                                       f'Clean_syn_C{args.drop_num}_{args.valid_ratio}.pt'.replace(" ", ""))

        if use_cache and os.path.exists(clean_dataset_split_path):
            print('Load cache datas')
            clean_dataset_info = torch.load(clean_dataset_split_path)

            self.sp_train = clean_dataset_info['train']
            self.sp_valid = clean_dataset_info['valid']
            self.sp_test = clean_dataset_info['test']
            self.num_users = clean_dataset_info['num_users']
            self.num_items = clean_dataset_info['num_items']

            self.df_user_pref_train = clean_dataset_info['user_pref_train']
            self.df_user_pref_train_valid = clean_dataset_info['user_pref_train_valid']
            self.df_user_pref_valid = clean_dataset_info['user_pref_valid']
            self.df_user_pref_test = clean_dataset_info['user_pref_test']
            self.num_cate = clean_dataset_info['num_cate']
            self.density = clean_dataset_info['density']
            self.num_interaction = clean_dataset_info['num_interaction']
            self.matrix_F = clean_dataset_info['matrix_F']
            self.item_category = clean_dataset_info['item_category']

            print('Done')
            print(
                f'num user: {self.num_users},',
                f'num item: {self.num_items},',
                f'num cate: {self.num_cate},',
                f'interaction: {self.num_interaction},',
                f'density: {self.density:.4f}')
        else:
            # set column name of data frame
            self.str_user, self.str_item, self.str_rating, self.str_time, self.str_cate, self.str_user_pref = args.str_cols
            clean_df_path = os.path.join(dir_path, f'clean_df_C{args.drop_num}.pt')
            if os.path.exists(clean_df_path):
                print(f'Load {self.dataset_name} data frame: {clean_df_path}')
                df_dict = torch.load(clean_df_path)
                df_clean = df_dict['clean_df']
                self.num_cate = df_dict['num_cate']
            else:
                file_path = os.path.join(dir_path, f'{args.file_name}')
                print('Read interaction datas')
                df = pd.read_csv(file_path, sep=args.sep, names=args.str_cols)
                df[self.str_cate] = df[self.str_cate].apply(ast.literal_eval)
                print('Done')

                le_user = LabelEncoder()
                le_item = LabelEncoder()
                print('Make clean datasets')
                # Drop and Sort interactions chronologically.
                df_clean = self.clean_and_sort(df, args.drop_num, args.drop_rating, le_user, le_item)
                # Encoding category information
                df_clean = self.enc_cate(df_clean)
                print('Done')

                print('Store clean dataframe as clean_df.pt')
                df_dict = {
                    'clean_df': df_clean,
                    'user_enc': le_user,
                    'item_enc': le_item,
                    'num_cate': self.num_cate,
                    'enc_dict': self.enc_dict,
                }
                torch.save(df_dict, clean_df_path)
                print('Done')

            self.num_users = df_clean[self.str_user].nunique()
            self.num_items = df_clean[self.str_item].nunique()
            self.density = df_clean.shape[0] / (self.num_users * self.num_items)
            self.num_interaction = df_clean.shape[0]

            clean_pref = self.make_user_cate_pref(df_clean)
            clean_test_cate_idx = [0] * self.num_users

            for idx, tmp_data in clean_pref.iterrows():
                tmp_prob = tmp_data[self.str_user_pref]
                non_zero_cnt = len(np.where(tmp_prob != 0)[0])
                cnt_small_indices = round(non_zero_cnt * 0.3)

                small_indices = []
                sorted_idx = np.argsort(tmp_prob)
                small_order = 0
                while len(small_indices) < cnt_small_indices:
                    if tmp_prob[sorted_idx[small_order]] == 0:
                        small_order += 1
                    else:
                        small_indices.append(sorted_idx[small_order])
                        small_order += 1

                clean_test_cate_idx[tmp_data[self.str_user]] = small_indices

            print(f'Split datasets as train, valid, test')
            df_train, df_valid, df_test = self.split_group_by_user(df_clean, args.str_cols, clean_test_cate_idx, args.valid_ratio)
            print('Done')

            # Encode due to the dropped users
            new_le_user = LabelEncoder()
            df_train[self.str_user] = new_le_user.fit_transform(df_train[self.str_user])
            df_valid[self.str_user] = new_le_user.transform(df_valid[self.str_user])
            df_test[self.str_user] = new_le_user.transform(df_test[self.str_user])

            assert df_train[self.str_user].nunique() == df_valid[self.str_user].nunique() == df_test[
                self.str_user].nunique(), 'Users not matching'

            self.num_users = df_train[self.str_user].nunique()

            print("Calculate user's category preferences")
            self.df_user_pref_train = self.make_user_cate_pref(df_train)
            self.df_user_pref_train_valid = self.make_user_cate_pref(pd.concat([df_train, df_valid]))
            self.df_user_pref_valid = self.make_user_cate_pref(df_valid)
            self.df_user_pref_test = self.make_user_cate_pref(df_test)
            print('Done')

            print(
                f'num user: {self.num_users},',
                f'num item: {self.num_items},',
                f'num cate: {self.num_cate},',
                f'interaction: {self.num_interaction},',
                f'density: {self.density:.4f}')

            print(f'Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}')
            self.sp_train = self.make_csr_matrix(df_train)
            self.sp_valid = self.make_csr_matrix(df_valid)
            self.sp_test = self.make_csr_matrix(df_test)

            print('Make encoded item-category pair')
            self.matrix_F = self.make_cate_multihot_matrix_F(df_clean)
            print('Done')

            print(f'Store datas')
            info_dict = {
                'train': self.sp_train,
                'valid': self.sp_valid,
                'test': self.sp_test,
                'user_pref_train': self.df_user_pref_train,
                'user_pref_train_valid': self.df_user_pref_train_valid,
                'user_pref_valid': self.df_user_pref_valid,
                'user_pref_test': self.df_user_pref_test,
                'num_users': self.num_users,
                'num_items': self.num_items,
                'density': self.density,
                'num_interaction': self.num_interaction,
                'num_cate': self.num_cate,
                'matrix_F': self.matrix_F,
                'item_category': self.item_category,
            }
            torch.save(info_dict, clean_dataset_split_path)
            print('Done')

        print(f'Data ready')

    def enc_cate(self, df_clean):
        all_cate = []
        df_clean_cate = df_clean.drop_duplicates(subset=self.str_item)[[self.str_item, self.str_cate]]
        for cate in df_clean_cate[self.str_cate]:
            all_cate.extend(cate)

        unique_cate = set(all_cate)
        self.num_cate = len(unique_cate)

        enc_dict = {}
        for idx, cate in enumerate(sorted(unique_cate)):
            enc_dict[cate] = idx
        self.enc_dict = enc_dict

        def apply_enc_cate(cate_list, enc_dict):
            # make category column datas to be list type
            ret_list = []
            for cate in sorted(cate_list):
                ret_list.append(enc_dict[cate])
            return ret_list

        df_clean[self.str_cate] = df_clean[self.str_cate].apply(apply_enc_cate, args=(enc_dict,))
        return df_clean

    def clean_and_sort(self, df, drop_num, drop_rating, le_user, le_item):
        def drop_unreliable(df, drop_rating):
            print('  Drop unreliable interaction')
            df_clean = df[df[self.str_rating] >= drop_rating]
            return df_clean

        def drop_unactive(df, str_col, drop_num):
            if str_col == self.str_user:
                df_group_size = df.groupby([self.str_user]).size()
            else:
                df_group_size = df.groupby([self.str_item]).size()

            clean_idx = df_group_size[df_group_size >= drop_num].index
            df_clean = df[df[str_col].isin(clean_idx)]
            return df_clean

        def is_unactive(df, str_col, drop_num):

            if str_col == self.str_user:
                df_group_size = df.groupby([self.str_user]).size()
            else:
                df_group_size = df.groupby([self.str_item]).size()

            unactive_df = df_group_size[df_group_size < drop_num].index
            print(f"    # of unactive interactions ({str_col}): {len(unactive_df)}")
            if len(unactive_df) == 0:
                return False  # False if unactive is None
            else:
                return True

        def is_unactive_cate(df, drop_num):
            all_cate = []
            for cate in df[self.str_cate]: all_cate.extend(cate)
            unique_cate = set(all_cate)
            selected_cates = [cate for cate in unique_cate if all_cate.count(cate) >= drop_num]

            print(f"    # of unactive categories: {len(unique_cate) - len(selected_cates)}")
            if (len(unique_cate) - len(selected_cates)) == 0:
                return False, selected_cates
            else:
                return True, selected_cates

        def drop_unactive_cate(df):
            while True:
                df_cate = df.drop_duplicates(subset=self.str_item)[[self.str_item, self.str_cate]]
                is_unact, selected_cates = is_unactive_cate(df_cate, drop_num)
                if is_unact is False: break
                df = df[df[self.str_cate].apply(lambda x: all(item in selected_cates for item in x))]

            return df

        def core_setting(df, drop_num):
            print('  Drop unactive users and items.')

            break_flag = False
            while True:
                if is_unactive(df, self.str_user, drop_num) is False:
                    df = drop_unactive_cate(df)
                    if is_unactive(df, self.str_user, drop_num) is False and is_unactive(df, self.str_item,
                                                                                         drop_num) is False:
                        break_flag = True
                else:
                    df = drop_unactive(df, self.str_user, drop_num)

                if break_flag is True: break

                if is_unactive(df, self.str_item, drop_num) is False:
                    df = drop_unactive_cate(df)
                    if is_unactive(df, self.str_user, drop_num) is False and is_unactive(df, self.str_item,
                                                                                         drop_num) is False:
                        break_flag = True
                else:
                    df = drop_unactive(df, self.str_item, drop_num)

                if break_flag is True: break

            return df

        # Drop duplicated (user, tiem)
        df = df.drop_duplicates(subset=[self.str_user, self.str_item]).reset_index(drop=True)
        if drop_rating:
            df = drop_unreliable(df, drop_rating)
        if drop_num:
            df = core_setting(df, drop_num)
        df[self.str_rating] = 1.0
        if self.dataset_name in 'anime':
            # random shuffle
            df = df.sample(frac=1).reset_index(drop=True)
            df_sorted = df.sort_values([self.str_user])
        else:
            # sort chronologically
            df_sorted = df.sort_values([self.str_user, self.str_time])

        # Encode user/item id
        df_sorted[self.str_user] = le_user.fit_transform(df_sorted[self.str_user])
        df_sorted[self.str_item] = le_item.fit_transform(df_sorted[self.str_item])
        return df_sorted

    def make_cate_multihot_matrix_F(self, df):
        # make item, category dictionary
        df_pair = df.drop_duplicates(subset=[self.str_item]).reset_index(drop=True)[[self.str_item, self.str_cate]]
        df_pair_sorted = df_pair.sort_values([self.str_item]).reset_index(drop=True)
        item_category = df_pair_sorted.set_index(self.str_item)[self.str_cate].to_dict()

        self.item_category = item_category

        # make matrix F
        matrix_F = [[0.] * self.num_items for _ in range(self.num_cate)]
        for item_id, item_category in item_category.items():
            for cate in item_category:
                matrix_F[cate][item_id] = 1 / len(item_category)
        return matrix_F

    def split_group_by_user(self, df, str_cols, test_cate, valid_ratio):
        group_users = df.groupby(self.str_user)

        train = []
        valid = []
        test = []

        for user_id, df_user in tqdm(group_users):
            cate_idx = test_cate[user_id]
            tmp_test_df = df_user[df_user[self.str_cate].apply(
                lambda x: True if set(cate_idx) & set(x) else False)]
            tmp_train_valid_df = df_user[df_user[self.str_cate].apply(
                lambda x: False if set(cate_idx) & set(x) else True)]
            tmp_train_valid_df = tmp_train_valid_df.sample(frac=1).reset_index(drop=True)

            np_data = tmp_train_valid_df.values
            num_item = len(tmp_train_valid_df)
            num_valid = np.ceil(num_item * valid_ratio).astype(int)

            if len(np_data[:-num_valid]) and len(np_data[-num_valid:]) and len(tmp_test_df.values):
                # split 못 하는 거 즉, train/valid/test에 data가 모두 있지 못한 user는 drop
                train.extend(np_data[:-num_valid])
                valid.extend(np_data[-num_valid:])
                test.extend(tmp_test_df.values)

        np_train = np.array(train)
        np_valid = np.array(valid)
        np_test = np.array(test)

        df_train = pd.DataFrame(data=np_train, columns=str_cols)
        df_valid = pd.DataFrame(data=np_valid, columns=str_cols)
        df_test = pd.DataFrame(data=np_test, columns=str_cols)

        return df_train, df_valid, df_test

    def make_user_cate_pref(self, df):
        def calculate_entropy(user_interactions, num_cate):
            cnt_cate_list = np.array([0.] * num_cate)
            for cate_list in user_interactions:
                for cate in cate_list:
                    cnt_cate_list[cate] += (1 / len(cate_list))

            return cnt_cate_list / cnt_cate_list.sum()

        df_grp_user = df.groupby(self.str_user)
        # Make the interacted category by user to be 2d np.array
        df_grp_user = df_grp_user[self.str_cate].apply(np.array).reset_index()
        cate_array = df_grp_user[self.str_cate].values

        # Calculate preference probability
        func_prob = np.frompyfunc(calculate_entropy, 2, 1)
        prob = func_prob(cate_array, self.num_cate)
        df_grp_user[self.str_user_pref] = prob
        return df_grp_user.drop('cate', axis=1)

    def make_csr_matrix(self, df):
        data = df[self.str_rating].values
        row = df[self.str_user].values
        col = df[self.str_item].values
        return sp.csr_matrix((data, (row, col)), dtype='float32', shape=(self.num_users, self.num_items))


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)  # cpu
    torch.cuda.manual_seed(random_seed)  # gpu
    np.random.seed(random_seed)  # numpy
    random.seed(random_seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing", add_help=True)

    ##### Setting #####
    parser.add_argument('--seed', default=1, type=int, help="Random seed")
    parser.add_argument('--dataset_name', default='ml-1m', type=str, help="Dataset name")
    parser.add_argument('--str_cols', default=['user', 'item', 'rating', 'timestamp', 'cate', 'user_pref'],
                        type=str, nargs="+", help="Interaction dataframe column names")
    parser.add_argument('--file_name', default='data.csv', type=str, help="Interaction file name")
    parser.add_argument('--drop_num', default=20, type=int, help="Drop user whose history are less than drop_num")
    parser.add_argument('--drop_rating', default=4, type=int,
                        help="Drop interaction of which rating is less than drop_rating")
    parser.add_argument('--valid_ratio', default=0.2, type=float, help="Valid ratio")
    parser.add_argument('--sep', default=',', type=str, help="Seperator of interaction csv file")

    args = parser.parse_args()
    set_random_seed(random_seed=args.seed)
    dir_path = os.path.join(os.getcwd(), 'dataset', args.dataset_name)
    print(f'Preprocess {args.dataset_name}: Core-{args.drop_num} setting')
    dataset = PreProcess(args, dir_path, use_cache=False)

