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

    dataset, train_dataset, valid_dataset, test_dataset, matrix_F = load_data(args, dataset_dir_path)
    
    from preprocessing import get_user_gender_dict
    user_gender_map = get_user_gender_dict(dataset_dir_path)
    
    sp_train, sp_valid, sp_test = dataset.sp_train, dataset.sp_valid, dataset.sp_test

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_recall, best_epoch = -np.inf, np.inf
    best_test_result, best_ckl = None, None
    start_epoch = 1

    print("Start training")

    start_total = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch - best_epoch >= args.patient:  # early stopping
            print('-' * 18)
            print('Exiting from training early')
            break

        start = time.time()
        loss = train_one_epoch(args, model, diffusion, optimizer, train_loader, matrix_F)
        print(f'Epoch {epoch:>3} - train loss: {loss: >10.4f}. time: {str(timedelta(seconds=int(time.time() - start)))}')

        if epoch % args.eval_freq == 0:
            metrics_all = evaluate(args, model, diffusion, valid_loader, sp_valid, sp_train, args.topK,
                                  dataset.item_category, dataset.num_cate, 1.0, user_gender_map)
            
            val_recall = metrics_all[2][1]
            
            print(f'Evaluation validation recall@20: {val_recall:.4f}, time: {str(timedelta(seconds=int(time.time() - start)))}')

            if val_recall > best_recall:
                best_recall = val_recall
                test_results = evaluate(args, model, diffusion, test_loader, sp_test, sp_train + sp_valid,
                                             args.topK, dataset.item_category, dataset.num_cate, 1.0, user_gender_map, is_best=True)

                best_epoch, best_test_result = epoch, test_results
                if args.save_model is True:
                    torch.save(model, best_model_path)
                print(f'Evaluation Best test')
                print_metric_results(args.topK, test_results)

    print('#' * 106, f'Training time: {str(timedelta(seconds=int(time.time() - start_total)))}')
    print(f"Test Metirc At Best Valid Metric, epoch: {best_epoch}")
    print_metric_results(args.topK, best_test_result)


def get_args_parser():
    parser = argparse.ArgumentParser(description="D3Rec", add_help=True)

    ##### Training Setting #####
    parser.add_argument('--seed', default=1, type=int, help="Random seed")
    parser.add_argument('--cuda', default=0, type=int, help="GPU index")
    parser.add_argument('--batch_size', default=400, type=int, help="Batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--wd', default=0.1, type=float, help="Weight decay")
    parser.add_argument('--epochs', default=2000, type=int, help="Num epochs")
    parser.add_argument('--topK', default=[10, 20], type=int, nargs='+', help="Top k list")
    parser.add_argument('--patient', default=1000, type=int, help="Early stop patient.")
    parser.add_argument('--save_model', action='store_true', help="Save model parameters.")
    parser.add_argument('--eval_freq', type=int, default=2)

    ##### Data Setting #####
    parser.add_argument('--dataset_name', default='ml-1m', type=str, help="Dataset name")
    parser.add_argument('--file_name', default='ratings.dat', type=str, help="Interaction file name")
    parser.add_argument('--sep', default='::', type=str, help="Separator of interaction file")
    parser.add_argument('--str_cols', default=['user', 'item', 'rating', 'timestamp', 'cate', 'user_pref'],
                        type=str, nargs="+", help="Column names")
    parser.add_argument('--split_ratio', default=[6, 2, 2], type=int, nargs="+", help="Train, Valid, Test split ratio")
    parser.add_argument('--drop_num', default=5, type=int, help="Min interactions")
    parser.add_argument('--test_w_valid', action='store_true',
                        help="True: test with train and valid data.")

    ##### Model hyper parameter #####
    parser.add_argument('--dims', type=int, default=[600, 200], nargs="+")
    parser.add_argument('--dropout', default=0.5, type=float, help="Drop interaction")
    parser.add_argument('--dim_step', default=10, type=int, help="Dimension of denoising step")
    parser.add_argument('--lamda', default=10.0, type=float, help="Penalty of Orthogonal and Matching loss")
    parser.add_argument('--w_max', default=10.0, type=float, help="Range of reweight")
    parser.add_argument('--w_min', default=0.2, type=float, help="Range of reweight")

    ### Diffusion hyper parameter
    parser.add_argument('--beta_start', default=0.0001, type=float)
    parser.add_argument('--beta_end', default=0.02, type=float)
    parser.add_argument('--noise_scale', default=0.1, type=float)
    parser.add_argument('--steps', default=100, type=int)
    parser.add_argument('--snr', action='store_true')
    parser.add_argument('--sampling_steps', default=0, type=int)
    parser.add_argument('--sampling_noise', action='store_true')
    parser.add_argument('--noise_schedule', default="linear-var", type=str,
                        choices=['linear', 'linear-var', 'cosine', 'exp', 'binomial', 'sqrt'])

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