import os
import torch
import numpy as np
from preprocessing import PreProcess
from torch.utils.data import Dataset

class D3RecData(Dataset):
    def __init__(self, sp_data, prob_in_df, prob_pred_df):
        """
        sp_data: Sparse matrix (etkileşimler)
        prob_in_df: Giriş profili DataFrame (user_pref sütunu içerir)
        prob_pred_df: Hedef profili DataFrame (user_pref sütunu içerir)
        """
        self.sp_data = sp_data
        self.prob_in = prob_in_df
        self.prob_pred = prob_pred_df

    def __getitem__(self, idx):
        # Etkileşim vektörünü al
        item = torch.FloatTensor(self.sp_data.getrow(idx).toarray()[0])
        
        # Kullanıcı profillerini güvenli şekilde çek (iloc ile)
        # .values[0] veya doğrudan listeyi Tensor'a çeviriyoruz
        p_in = torch.Tensor(self.prob_in.iloc[idx]['user_pref']).to(torch.float32)
        p_pred = torch.Tensor(self.prob_pred.iloc[idx]['user_pref']).to(torch.float32)
        
        return item, p_in, p_pred

    def __len__(self):
        return self.sp_data.shape[0]

def load_data(args, dir_path):
    # PreProcess içindeki cache mekanizmasını kullanır
    dataset = PreProcess(args, dir_path, use_cache=True)

    # Training: Giriş ve Hedef aynı (Kendi profilini öğrensin)
    train_dataset = D3RecData(dataset.sp_train, dataset.df_user_pref_train, dataset.df_user_pref_train)
    
    # Validation: Giriş ve Hedef aynı
    valid_dataset = D3RecData(dataset.sp_valid, dataset.df_user_pref_valid, dataset.df_user_pref_valid)
    
    # Test: Giriş train (veya train+valid), Hedef ise test profili
    if args.test_w_valid:
        test_dataset = D3RecData(
            dataset.sp_train + dataset.sp_valid, 
            dataset.df_user_pref_train_valid, 
            dataset.df_user_pref_test
        )
    else:
        test_dataset = D3RecData(
            dataset.sp_train, 
            dataset.df_user_pref_train, 
            dataset.df_user_pref_test
        )

    # shape: (category, num_items)
    matrix_F = torch.tensor(np.array(dataset.matrix_F), dtype=torch.float32, device=args.device)

    return dataset, train_dataset, valid_dataset, test_dataset, matrix_F