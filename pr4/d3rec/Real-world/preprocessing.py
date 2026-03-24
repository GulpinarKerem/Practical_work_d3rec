import os
import ast
import random
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# ============================
# POPULARITY BIN GENERATOR

def generate_popularity_bins(df_inter, item_col="item_id", user_col="user_id",
                             bin_labels=("high", "mid", "low"),
                             bin_ratios=(0.3, 0.3, 0.4)):
    """
    Her item'ın popülaritesini hesaplar ve binlere ayırır.
    """
    item_pop = df_inter.groupby(item_col).size().reset_index(name="popularity")
    item_pop = item_pop.sort_values("popularity", ascending=False).reset_index(drop=True)

    total_pop = item_pop["popularity"].sum()
    item_pop["cum_ratio"] = item_pop["popularity"].cumsum() / max(total_pop, 1)

    high_edge = bin_ratios[0]
    mid_edge  = bin_ratios[0] + bin_ratios[1]

    item_pop["pop_bin"] = "low"
    item_pop.loc[item_pop["cum_ratio"] <= mid_edge, "pop_bin"] = bin_labels[1]
    item_pop.loc[item_pop["cum_ratio"] <= high_edge, "pop_bin"] = bin_labels[0]

    return item_pop[[item_col, "pop_bin"]]

def calculate_user_specific_prefs(df_inter, item_to_cat_map, num_cate=3):
    """
    Her kullanıcı için eğitim setindeki popülerlik dağılımını hesaplar.
    """
    user_prefs = {}
    user_groups = df_inter.groupby('user_id')
    
    for uid, group in user_groups:
        items = group['item_id'].values
        cats = [item_to_cat_map.get(it, [1])[0] for it in items] 
        
        total = len(cats)
        counts = np.zeros(num_cate)
        for c in cats:
            if isinstance(c, int):
                counts[c] += 1
        
        user_prefs[uid] = (counts / max(total, 1)).tolist()
        
    return user_prefs


class PreProcess():
    def __init__(self, args, dir_path, use_cache=False):
        self.dataset_name = args.dataset_name
        clean_dataset_split_path = os.path.join(dir_path,
                                       f'Clean_C{args.drop_num}_{args.split_ratio}.pt'.replace(" ", ""))

        if use_cache and os.path.exists(clean_dataset_split_path):
            print('Load cache datas')
            clean_dataset_info = torch.load(clean_dataset_split_path, weights_only=False)

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
        else:
            # Sütun isimlerini güvenli tanımla
            self.str_user = args.str_cols[0]
            self.str_item = args.str_cols[1]
            self.str_rating = args.str_cols[2]
            self.str_time = args.str_cols[3]
            self.str_cate = 'cate'
            self.str_user_pref = 'user_pref'

            clean_df_path = os.path.join(dir_path, f'clean_df_C{args.drop_num}.pt')
            
            if os.path.exists(clean_df_path):
                print(f'Load {self.dataset_name} data frame: {clean_df_path}')
                df_dict = torch.load(clean_df_path, weights_only=False)
                df_clean = df_dict['clean_df']
                self.num_cate = df_dict['num_cate']
            else:
                file_path = os.path.join(dir_path, f'{args.file_name}')
                print('Read interaction datas')
                df = pd.read_csv(file_path, sep=args.sep, names=args.str_cols[:4], engine='python')
                print('Done')

                le_user = LabelEncoder()
                le_item = LabelEncoder()
                print('Make clean datasets')
                # Kritik: Popülerlik binleri clean_and_sort içinde core_setting'den önce oluşturulacak
                df_clean = self.clean_and_sort(df, args.drop_num, args.drop_rating, le_user, le_item)
                
                # Encoding category information (pop_bin to int)
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

            print(f'Split datasets as train, valid, test: {args.split_ratio}')
            df_train, df_valid, df_test = self.split_group_by_user(df_clean, args.split_ratio, [self.str_user, self.str_item, self.str_rating, self.str_time, self.str_cate])
            print('Done')

            print("Calculate user's category preferences")
            self.df_user_pref_train = self.make_user_cate_pref(df_train)
            self.df_user_pref_train_valid = self.make_user_cate_pref(pd.concat([df_train, df_valid]))
            self.df_user_pref_valid = self.make_user_cate_pref(df_valid)
            self.df_user_pref_test = self.make_user_cate_pref(df_test)
            print('Done')

            self.num_users = df_clean[self.str_user].nunique()
            self.num_items = df_clean[self.str_item].nunique()
            self.density = df_clean.shape[0] / (self.num_users * self.num_items)
            self.num_interaction = df_clean.shape[0]

            self.sp_train = self.make_csr_matrix(df_train)
            self.sp_valid = self.make_csr_matrix(df_valid)
            self.sp_test = self.make_csr_matrix(df_test)

            print('Make encoded item-category pair')
            self.matrix_F = self.make_cate_multihot_matrix_F(df_clean)
            print('Done')

            info_dict = {
                'train': self.sp_train, 'valid': self.sp_valid, 'test': self.sp_test,
                'user_pref_train': self.df_user_pref_train,
                'user_pref_train_valid': self.df_user_pref_train_valid,
                'user_pref_valid': self.df_user_pref_valid,
                'user_pref_test': self.df_user_pref_test,
                'num_users': self.num_users, 'num_items': self.num_items,
                'density': self.density, 'num_interaction': self.num_interaction,
                'num_cate': self.num_cate, 'matrix_F': self.matrix_F,
                'item_category': self.item_category,
            }
            torch.save(info_dict, clean_dataset_split_path)
            print('Done')

    def enc_cate(self, df_clean):
        all_cate = []
        df_clean_cate = df_clean.drop_duplicates(subset=self.str_item)[[self.str_item, self.str_cate]]
        for cate in df_clean_cate[self.str_cate]:
            all_cate.extend(cate)

        unique_cate = set(all_cate)
        self.num_cate = len(unique_cate)
        enc_dict = {cate: idx for idx, cate in enumerate(sorted(unique_cate))}
        self.enc_dict = enc_dict

        df_clean[self.str_cate] = df_clean[self.str_cate].apply(lambda x: [enc_dict[c] for c in x])
        return df_clean

    def clean_and_sort(self, df, drop_num, drop_rating, le_user, le_item):
        # 1. Temel temizlik
        df = df.drop_duplicates(subset=[self.str_user, self.str_item]).reset_index(drop=True)
        if drop_rating:
            df = df[df[self.str_rating] >= drop_rating]

        # --- KRİTİK: BURADA CATE SÜTUNUNU OLUŞTURUYORUZ ---
        pop_bins = generate_popularity_bins(df, item_col=self.str_item, user_col=self.str_user)
        df = df.merge(pop_bins, on=self.str_item, how="left")
        df[self.str_cate] = df["pop_bin"].apply(lambda x: [x])
        
        # 2. Core setting (Artık cate sütunu var, çökmez)
        if drop_num:
            df = self.core_setting(df, drop_num)

        df[self.str_rating] = 1.0
        
        if self.dataset_name == 'anime':
            df = df.sample(frac=1).reset_index(drop=True)
            df_sorted = df.sort_values([self.str_user])
        else:
            df_sorted = df.sort_values([self.str_user, self.str_time])

        df_sorted[self.str_user] = le_user.fit_transform(df_sorted[self.str_user])
        df_sorted[self.str_item] = le_item.fit_transform(df_sorted[self.str_item])
        return df_sorted

    def core_setting(self, df, drop_num):
        print('  Drop unactive users and items.')
        while True:
            # User filtering
            user_counts = df.groupby(self.str_user).size()
            clean_users = user_counts[user_counts >= drop_num].index
            df = df[df[self.str_user].isin(clean_users)]
            
            # Item filtering
            item_counts = df.groupby(self.str_item).size()
            clean_items = item_counts[item_counts >= drop_num].index
            df = df[df[self.str_item].isin(clean_items)]
            
            if user_counts.min() >= drop_num and item_counts.min() >= drop_num:
                break
        return df

    def make_cate_multihot_matrix_F(self, df):
        df_pair = df.drop_duplicates(subset=[self.str_item]).sort_values(self.str_item)
        item_category = df_pair.set_index(self.str_item)[self.str_cate].to_dict()
        self.item_category = item_category

        matrix_F = [[0.] * self.num_items for _ in range(self.num_cate)]
        for item_id, cats in item_category.items():
            for cate in cats:
                matrix_F[cate][item_id] = 1 / len(cats)
        return matrix_F

    def split_group_by_user(self, df, ratio, str_cols):
        sum_ratio = sum(ratio)
        test_ratio = ratio[2] / sum_ratio
        val_ratio = ratio[1] / (ratio[0] + ratio[1])

        train, valid, test = [], [], []
        for user_id, df_user in tqdm(df.groupby(self.str_user)):
            n = len(df_user)
            n_test = int(np.ceil(n * test_ratio))
            n_val = int(np.ceil((n - n_test) * val_ratio))
            n_train = n - n_test - n_val
            
            user_data = df_user.values
            train.extend(user_data[:n_train])
            valid.extend(user_data[n_train:n_train+n_val])
            test.extend(user_data[n_train+n_val:])

        
        return pd.DataFrame(train, columns=df.columns), pd.DataFrame(valid, columns=df.columns), pd.DataFrame(test, columns=df.columns)

    def make_user_cate_pref(self, df):
        def calc_pref(user_interactions, num_cate):
            cnt = np.zeros(num_cate)
            for cats in user_interactions:
                for c in cats: cnt[c] += (1 / len(cats))
            return (cnt / (cnt.sum() + 1e-7)).tolist()

        df_grp = df.groupby(self.str_user)[self.str_cate].apply(list).reset_index()
        df_grp[self.str_user_pref] = df_grp[self.str_cate].apply(calc_pref, args=(self.num_cate,))
        return df_grp.drop(self.str_cate, axis=1)

    def make_csr_matrix(self, df):
        return sp.csr_matrix((df[self.str_rating], (df[self.str_user], df[self.str_item])),
                             shape=(self.num_users, self.num_items))

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

def get_user_gender_dict(dataset_dir_path):
    import os
    path = os.path.join(dataset_dir_path, 'users.dat')
    gender_dict = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2: gender_dict[int(parts[0])] = parts[1]
        print(f"--- SUCCESS: Gender map loaded. Total: {len(gender_dict)}")
    return gender_dict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--dataset_name', default='ml-1m', type=str)
    parser.add_argument('--str_cols', default=['user', 'item', 'rating', 'timestamp', 'cate', 'user_pref'], type=str, nargs="+")
    parser.add_argument('--file_name', default='ratings.dat', type=str)
    parser.add_argument('--drop_num', default=5, type=int)
    parser.add_argument('--drop_rating', default=4, type=int)
    parser.add_argument('--split_ratio', default=[6, 2, 2], type=int, nargs="+")
    parser.add_argument('--sep', default='::', type=str)
    args = parser.parse_args()
    set_random_seed(args.seed)
    dir_path = os.path.join(os.getcwd(), 'dataset', args.dataset_name)
    PreProcess(args, dir_path, use_cache=False)