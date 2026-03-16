import os, time
from datetime import timedelta
import argparse
import numpy as np
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
    
    # Modelin varlığını kontrol et
    if not os.path.exists(best_model_path):
        print(f"HATA: Model dosyası bulunamadı! Yol: {best_model_path}")
        return

    # Veriyi yükle (use_cache=False yaparak güncel binleri aldığından emin oluyoruz)
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

    # Modeli yükleme
    print(f"Loading model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=args.device, weights_only=False).state_dict())
    model.eval()

    # =========================================================================
    # [DEBUG] REPRESENTATION SUCCESS TEST (PHASE 4 PROOF)
    # =========================================================================
    print("\n" + "="*50)
    print("[DEBUG] EXTREME TARGET TEST: target = [0, 0, 1] (Tail-Only)")
    
    num_debug_users = 50
    extreme_target = torch.tensor([[0.0, 0.0, 1.0]], device=args.device).repeat(num_debug_users, 1)

    with torch.no_grad():
        try:
            it = iter(loader)
            first_batch_x, _, _ = next(it)
            first_batch_x = first_batch_x[:num_debug_users].to(args.device)
            
            # İpucundan aldığımız gerçek metod adı: sample_new_interaction
            sampled_x = diffusion.sample_new_interaction(
                model, 
                first_batch_x, 
                extreme_target, 
                args.guide_w, 
                args.sampling_steps
            )
            
            # Zaten izlenenleri ele
            sampled_x[first_batch_x > 0] = -1e9
            
            _, top_indices = torch.topk(sampled_x, k=10, dim=-1)
            
            hist = np.zeros(3)
            for i in range(num_debug_users):
                for item_idx in top_indices[i]:
                    cate = dataset.item_category[item_idx.item()]
                    hist[cate] += 1

            share = hist / hist.sum()
            print(f"[DEBUG] K=10 first50users hist={hist.tolist()} share={share.tolist()}")
            
            if hist[2] > 400:
                print("Status: SUCCESS (Model can represent the tail perfectly)")
            else:
                print(f"Status: WARNING (Tail count: {hist[2]}/500 - Bias persists)")
        except Exception as e:
            print(f"[DEBUG ERROR] Histogram testi sırasında hata: {e}")
    print("="*50 + "\n")

    # Sıcaklık (Temperature) döngüsü
    for temperature in [0.1, 0.5, 1.0, 5.0, 10.0]:
        start = time.time()
        results = evaluate(args, model, diffusion, loader, sp_test, sp_train + sp_valid, args.topK, dataset.item_category, dataset.num_cate, temperature, is_best=True)
        print(f'\n--- Evaluation with temperature {temperature} ---')
        print_metric_results(args.topK, results)
        print(f'Time: {str(timedelta(seconds=int(time.time() - start)))}')

def get_args_parser():
    parser = argparse.ArgumentParser(description="D3Rec", add_help=True)

    ##### Training Setting #####
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--topK', default=[10, 20], type=int, nargs='+')

    ##### Data Setting #####
    parser.add_argument('--dataset_name', default='ml-1m', type=str)
    parser.add_argument('--drop_num', default=5, type=int)
    parser.add_argument('--str_cols', default=['user', 'item', 'rating', 'timestamp', 'cate', 'user_pref'], type=str, nargs="+")
    parser.add_argument('--file_name', default='data.csv', type=str)
    parser.add_argument('--sep', default=',', type=str)
    parser.add_argument('--drop_rating', default=4, type=int)
    parser.add_argument('--split_ratio', default=[6, 2, 2], type=int, nargs="+")
    parser.add_argument('--test_w_valid', action='store_true')

    ##### Model hyper parameter #####
    parser.add_argument('--dims', type=int, default=[600, 200], nargs="+")
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--dim_step', default=10, type=int)
    parser.add_argument('--lamda', default=5.0, type=float)
    parser.add_argument('--w_max', default=1.0, type=float)
    parser.add_argument('--w_min', default=0.2, type=float)

    ### Diffusion hyper parameter
    parser.add_argument('--beta_start', default=0.0001, type=float)
    parser.add_argument('--beta_end', default=0.02, type=float)
    parser.add_argument('--noise_scale', default=0.1, type=float)
    parser.add_argument('--steps', default=100, type=int)
    parser.add_argument('--snr', action='store_true')
    parser.add_argument('--sampling_steps', default=0, type=int)
    parser.add_argument('--sampling_noise', action='store_true')
    parser.add_argument('--noise_schedule', default="linear-var", type=str)

    ### Classifier-free
    parser.add_argument('--drop_div', default=0.1, type=float)
    parser.add_argument('--guide_w', default=2.0, type=float)

    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    set_random_seed(random_seed=args.seed)
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    
    dataset_dir_path = os.path.join(os.getcwd(), 'dataset', args.dataset_name)
    
    # Model yolunu r"..." formatında tutmak Windows yolları için en güvenlisi
    best_model_path = r"Y:\Desktop\phase34\d3rec\Real-world\Best_models-C5-[6,2,2]\ml-1m\best_model.pt"

    main(args, dataset_dir_path, best_model_path)