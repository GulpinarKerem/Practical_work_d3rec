import os
import ast
import random
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def generate_popularity_bins(df_inter, item_col="item_id", user_col="user_id",
                             bin_labels=("high", "mid", "low"),
                             bin_ratios=(0.3, 0.3, 0.4)):
    # Item popülerlik sayısını hesapla
    item_pop = df_inter.groupby(item_col).size().reset_index(name="popularity")
    # Popülerliğe göre büyükten küçüğe sırala
    item_pop = item_pop.sort_values("popularity", ascending=False).reset_index(drop=True)

    total_pop = item_pop["popularity"].sum()
    item_pop["cum_ratio"] = item_pop["popularity"].cumsum() / max(total_pop, 1)

    high_edge = bin_ratios[0]
    mid_edge  = bin_ratios[0] + bin_ratios[1]

    item_pop["pop_bin"] = "low"
    item_pop.loc[item_pop["cum_ratio"] <= mid_edge, "pop_bin"] = bin_labels[1]
    item_pop.loc[item_pop["cum_ratio"] <= high_edge, "pop_bin"] = bin_labels[0]

    return item_pop[[item_col, "pop_bin"]]

def get_user_gender_dict(dataset_dir_path):
    possible_paths = [
        os.path.join(dataset_dir_path, 'users.dat'),
        os.path.join(os.path.dirname(dataset_dir_path), 'users.dat'),
        os.path.join(os.getcwd(), 'users.dat')
    ]
    
    users_file = None
    for path in possible_paths:
        if os.path.exists(path):
            users_file = path
            break

    gender_dict = {}
    if users_file:
        try:
            with open(users_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 2:
                        uid = int(parts[0])
                        gender = parts[1]
                        gender_dict[uid] = gender
            print(f"--- SUCCESS: Gender map loaded from {users_file}. Total: {len(gender_dict)}")
        except Exception as e:
            print(f"--- ERROR reading users.dat: {e}")
    else:
        print(f"--- WARNING: users.dat NOT FOUND. Gender audit will fail.")
        
    return gender_dict

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
            # Gender bilgisini cache'den çekiyoruz
            self.user_gender = clean_dataset_info.get('user_gender', {})

            print('Done')
        else:
            self.str_user, self.str_item, self.str_rating, self.str_time, self.str_cate, self.str_user_pref = args.str_cols
            clean_df_path = os.path.join(dir_path, f'clean_df_C{args.drop_num}.pt')
            
            if os.path.exists(clean_df_path):
                print(f'Load {self.dataset_name} data frame: {clean_df_path}')
                df_dict = torch.load(clean_df_path, weights_only=False)
                df_clean = df_dict['clean_df']
                self.num_cate = df_dict['num_cate']
            else:
                file_path = os.path.join(dir_path, f'{args.file_name}')
                
                print('Read interaction datas')
                # MovieLens 1M için sadece ilk 4 sütunu oku, cate ve user_pref'i biz sonra ekleyeceğiz
                df = pd.read_csv(file_path, sep=args.sep, names=args.str_cols[:4], engine='python')
                
                # Olmayan sütunları geçici olarak boş oluştur
                df[self.str_cate] = "[]" 
                df[self.str_user_pref] = 0.0
                
                print('Done')

                le_user = LabelEncoder()
                le_item = LabelEncoder()
                df_clean = self.clean_and_sort(df, args.drop_num, args.drop_rating, le_user, le_item)
                df_clean = self.enc_cate(df_clean)

                df_dict = {'clean_df': df_clean, 'user_enc': le_user, 'item_enc': le_item, 
                           'num_cate': self.num_cate, 'enc_dict': self.enc_dict}
                torch.save(df_dict, clean_df_path)

            df_train, df_valid, df_test = self.split_group_by_user(df_clean, args.split_ratio, args.str_cols)

            self.df_user_pref_train = self.make_user_cate_pref(df_train)
            self.df_user_pref_train_valid = self.make_user_cate_pref(pd.concat([df_train, df_valid]))
            self.df_user_pref_valid = self.make_user_cate_pref(df_valid)
            self.df_user_pref_test = self.make_user_cate_pref(df_test)

            self.num_users = df_clean[self.str_user].nunique()
            self.num_items = df_clean[self.str_item].nunique()
            self.density = df_clean.shape[0] / (self.num_users * self.num_items)
            self.num_interaction = df_clean.shape[0]

            self.sp_train = self.make_csr_matrix(df_train)
            self.sp_valid = self.make_csr_matrix(df_valid)
            self.sp_test = self.make_csr_matrix(df_test)

            self.matrix_F = self.make_cate_multihot_matrix_F(df_clean)
            
            # Cinsiyet bilgilerini yüklüyoruz
            print('Processing gender information...')
            self.user_gender = get_user_gender_dict(dir_path)

            info_dict = {
                'train': self.sp_train, 'valid': self.sp_valid, 'test': self.sp_test,
                'user_pref_train': self.df_user_pref_train, 'user_pref_train_valid': self.df_user_pref_train_valid,
                'user_pref_valid': self.df_user_pref_valid, 'user_pref_test': self.df_user_pref_test,
                'num_users': self.num_users, 'num_items': self.num_items,
                'density': self.density, 'num_interaction': self.num_interaction,
                'num_cate': self.num_cate, 'matrix_F': self.matrix_F,
                'item_category': self.item_category,
                'user_gender': self.user_gender # Cache'e kaydediyoruz
            }
            torch.save(info_dict, clean_dataset_split_path)
            print('Data ready and cached.')

    def enc_cate(self, df_clean):
        all_cate = []
        df_clean_cate = df_clean.drop_duplicates(subset=self.str_item)[[self.str_item, self.str_cate]]
        for cate in df_clean_cate[self.str_cate]:
            all_cate.extend(cate)
        unique_cate = set(all_cate)
        self.num_cate = len(unique_cate)
        enc_dict = {cate: idx for idx, cate in enumerate(sorted(unique_cate))}
        self.enc_dict = enc_dict
        df_clean[self.str_cate] = df_clean[self.str_cate].apply(lambda x: [enc_dict[c] for c in sorted(x)])
        return df_clean

    def clean_and_sort(self, df, drop_num, drop_rating, le_user, le_item):
        # 1. Tekrarları sil
        df = df.drop_duplicates(subset=[self.str_user, self.str_item]).reset_index(drop=True)
        
        # 2. Düşük ratingleri at
        if drop_rating: 
            df = df[df[self.str_rating] >= drop_rating]
        
        # 3. Popülerlik binlerini hesapla (Araştırma için 3-3-4 dağılımı)
        pop_bins = generate_popularity_bins(df, 
                                            item_col=self.str_item, 
                                            user_col=self.str_user,
                                            bin_labels=("high", "mid", "low"), 
                                            bin_ratios=(0.3, 0.3, 0.4))
        
        # 4. Binleri ana tabloya ekle
        df = df.merge(pop_bins, on=self.str_item, how="left")
        
        # 5. Bin bilgisini 'cate' sütununa yaz (Modelin anlayacağı format)
        df[self.str_cate] = df["pop_bin"].apply(lambda x: [x])

        # !!! KRİTİK NOKTA: Fazla olan pop_bin sütununu burada siliyoruz ki 6 sütun kalsın
        if "pop_bin" in df.columns:
            df = df.drop(columns=["pop_bin"])

        # 6. Sırala ve Encode et
        df_sorted = df.sort_values([self.str_user, self.str_time])
        df_sorted[self.str_user] = le_user.fit_transform(df_sorted[self.str_user])
        df_sorted[self.str_item] = le_item.fit_transform(df_sorted[self.str_item])
        
        return df_sorted

    def make_cate_multihot_matrix_F(self, df):
        df_pair = df.drop_duplicates(subset=[self.str_item]).sort_values([self.str_item])
        self.item_category = df_pair.set_index(self.str_item)[self.str_cate].to_dict()
        matrix_F = [[0.] * self.num_items for _ in range(self.num_cate)]
        for item_id, cates in self.item_category.items():
            for cate in cates:
                matrix_F[cate][item_id] = 1 / len(cates)
        return matrix_F

    def split_group_by_user(self, df, ratio, str_cols):
        np_data = df.values
        group_users = df.groupby(self.str_user)
        sum_ratio = sum(ratio)
        test_ratio = round(ratio[2] / sum_ratio, 3)
        val_ratio = round(ratio[1] / (ratio[0] + ratio[1]), 3)

        num_cum_items = 0
        train, valid, test = [], [], []

        for _, df_user in tqdm(group_users):
            num_items = len(df_user)
            num_test = np.ceil(num_items * test_ratio).astype(int)
            num_valid = np.ceil((num_items - num_test) * val_ratio).astype(int)
            num_train = int(num_items - num_test - num_valid)

            train.extend(np_data[num_cum_items:num_cum_items + num_train, :])
            valid.extend(np_data[num_cum_items + num_train:num_cum_items + num_train + num_valid, :])
            test.extend(np_data[num_cum_items + num_train + num_valid:num_cum_items + num_train + num_valid + num_test, :])
            num_cum_items += num_items

        return pd.DataFrame(train, columns=str_cols), pd.DataFrame(valid, columns=str_cols), pd.DataFrame(test, columns=str_cols)

    def make_user_cate_pref(self, df):
        def calc_pref(user_interactions, num_cate):
            cnt = np.zeros(num_cate)
            for c_list in user_interactions:
                for c in c_list: cnt[c] += (1 / len(c_list))
            return cnt / (cnt.sum() + 1e-7)

        df_grp = df.groupby(self.str_user)[self.str_cate].apply(list).reset_index()
        df_grp[self.str_user_pref] = df_grp[self.str_cate].apply(calc_pref, args=(self.num_cate,))
        return df_grp.drop(self.str_cate, axis=1)

    def make_csr_matrix(self, df):
        return sp.csr_matrix((df[self.str_rating].values, (df[self.str_user].values, df[self.str_item].values)), 
                             shape=(self.num_users, self.num_items))

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True