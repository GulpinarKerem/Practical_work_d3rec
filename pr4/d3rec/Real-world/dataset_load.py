import os
import torch
import numpy as np
from torch.utils.data import Dataset

from preprocessing import PreProcess


class D3RecData(Dataset):
    def __init__(self, sp_data, prob_in_df, prob_pred_df):
        self.sp_data = sp_data
        self.prob_in = prob_in_df.reset_index(drop=True)
        self.prob_pred = prob_pred_df.reset_index(drop=True)

    def __getitem__(self, idx):
        item = torch.FloatTensor(self.sp_data.getrow(idx).toarray()[0])
        p_in = torch.tensor(self.prob_in.iloc[idx]["user_pref"], dtype=torch.float32)
        p_pred = torch.tensor(self.prob_pred.iloc[idx]["user_pref"], dtype=torch.float32)
        return item, p_in, p_pred

    def __len__(self):
        return self.sp_data.shape[0]


def load_data(args, dir_path):
    dataset = PreProcess(args, dir_path, use_cache=True)

    train_dataset = D3RecData(
        dataset.sp_train,
        dataset.df_user_pref_train,
        dataset.df_user_pref_train,
    )

    
    valid_dataset = D3RecData(
        dataset.sp_train,
        dataset.df_user_pref_train,
        dataset.df_user_pref_valid,
    )

    if args.test_w_valid:
        test_dataset = D3RecData(
            dataset.sp_train + dataset.sp_valid,
            dataset.df_user_pref_train_valid,
            dataset.df_user_pref_test,
        )
    else:
        test_dataset = D3RecData(
            dataset.sp_train,
            dataset.df_user_pref_train,
            dataset.df_user_pref_test,
        )

    matrix_F = torch.tensor(np.array(dataset.matrix_F), dtype=torch.float32, device=args.device)

    return dataset, train_dataset, valid_dataset, test_dataset, matrix_F


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data loader", add_help=True)
    parser.add_argument("--cuda", default=0, type=int, help="GPU index")
    parser.add_argument("--dataset_name", default="ml-1m", type=str, help="Dataset name")
    parser.add_argument("--test_w_valid", action="store_true")
    parser.add_argument("--drop_num", default=5, type=int)
    parser.add_argument("--split_ratio", default=[6, 2, 2], type=int, nargs="+")
    parser.add_argument(
        "--str_cols",
        default=["user", "item", "rating", "timestamp", "cate", "user_pref"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--file_name", default="ratings.dat", type=str)
    parser.add_argument("--drop_rating", default=4, type=int)
    parser.add_argument("--sep", default="::", type=str)

    args = parser.parse_args()

    dir_path = os.path.join(os.getcwd(), "dataset", args.dataset_name)
    args.device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"Use: {args.device}")

    dataset, train_dataset, valid_dataset, test_dataset, matrix_F = load_data(args, dir_path)
    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))
    print("Test samples :", len(test_dataset))