import os, time
from datetime import timedelta
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset_load import load_data
from utils_train import *
from utils import *
from models import *

def main(args, dataset_dir_path, best_model_path, user_gender_map):
    print(args)
    print(f'Use {args.device}')
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # dataset_load.py içindeki load_data args içindeki test_w_valid vb. her şeyi bekler
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

    # Modeli yükle
    model.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=False).state_dict())

    # Sıcaklık döngüsü
    for temperature in [0.1, 0.2, 0.5, 1.0, 5.0]:
        start = time.time()
        results = evaluate_with_gender(args, model, diffusion, loader, sp_test, dataset, temperature, user_gender_map)
        
        print(f'Evaluation with w: {args.guide_w}, temperature {temperature} time: {str(timedelta(seconds=int(time.time() - start)))}')

def evaluate_with_gender(args, model, diffusion, loader, sp_test, dataset, temperature, user_gender_map):
    model.eval()
    
    # Tüm senaryolar için veri kapları
    res = {
        'F_Natural': {'target': [], 'pred': []},
        'F_Guided':  {'target': [], 'pred': []},
        'M_Natural': {'target': [], 'pred': []},
        'M_Guided':  {'target': [] , 'pred': []},
        'ALL_Natural': {'target': [], 'pred': []},
        'ALL_Guided':  {'target': [], 'pred': []}
    }
    
    all_targets = []
    for u in range(sp_test.shape[0]):
        all_targets.append(sp_test.getrow(u).indices.tolist())

    # --- HATA BURADAYDI: current_user_idx mutlaka burada (döngü dışında) tanımlanmalı ---
    current_user_idx = 0 
    item_category = dataset.item_category
    n_cate = dataset.num_cate

    with torch.no_grad():
        for x_0, prob, prob_pred in loader:
            x_0, prob = x_0.to(args.device), prob.to(args.device)
            
            # 1. GEÇİŞ: DOĞAL (Modelin kendi tahmini)
            x_0_nat = diffusion.sample_new_interaction(model, x_0, prob, args.guide_w, args.sampling_steps)
            x_0_nat[x_0 > 0] = -np.inf
            preds_nat = torch.topk(x_0_nat, k=max(args.topK), dim=-1)[1].cpu().numpy().tolist()

            # 2. GEÇİŞ: MANİPÜLE (Düşük Popülerlik Hedefli)
            low_pop_prob = torch.tensor([0.1, 0.1, 0.8], device=args.device).repeat(x_0.shape[0], 1)
            low_pop_prob = adjust_div(low_pop_prob, temperature)
            x_0_gui = diffusion.sample_new_interaction(model, x_0, low_pop_prob, args.guide_w, args.sampling_steps)
            x_0_gui[x_0 > 0] = -np.inf
            preds_gui = torch.topk(x_0_gui, k=max(args.topK), dim=-1)[1].cpu().numpy().tolist()

            for i in range(len(x_0)):
                uid = current_user_idx + i + 1
                gender = user_gender_map.get(uid, "M")
                target = all_targets[current_user_idx + i]
                
                # Cinsiyet Bazlı Kayıt
                if gender == 'F':
                    res['F_Natural']['target'].append(target); res['F_Natural']['pred'].append(preds_nat[i])
                    res['F_Guided']['target'].append(target); res['F_Guided']['pred'].append(preds_gui[i])
                else:
                    res['M_Natural']['target'].append(target); res['M_Natural']['pred'].append(preds_nat[i])
                    res['M_Guided']['target'].append(target); res['M_Guided']['pred'].append(preds_gui[i])
                
                # GENEL TOPLAM Kayıt (Senin istediğin ekleme)
                res['ALL_Natural']['target'].append(target); res['ALL_Natural']['pred'].append(preds_nat[i])
                res['ALL_Guided']['target'].append(target); res['ALL_Guided']['pred'].append(preds_gui[i])

            current_user_idx += len(x_0)

    # --- RAPORLAMA BÖLÜMÜ ---
    print(f"\n" + "="*20 + f" TAM KAPSAMLI ANALİZ (Temp: {temperature}) " + "="*20)
    
    def print_res(name, data):
        if not data['target']: return
        m = compute_metric(data['target'], data['pred'], args.topK, item_category, n_cate)
        print(f"\n>>> {name}:")
        print_metric_results(args.topK, m)

    # 1. Genel tabloyu en başa koyalım ki ana farkı görelim
    print_res("SİSTEM GENELİ - DOĞAL (Kontrol)", res['ALL_Natural'])
    print_res("SİSTEM GENELİ - MANİPÜLE (Low Pop)", res['ALL_Guided'])
    print("\n" + "-"*60)
    
    # 2. Cinsiyet detayları
    print_res("KADINLAR - DOĞAL", res['F_Natural'])
    print_res("KADINLAR - MANİPÜLE", res['F_Guided'])
    print("\n" + "-"*30)
    print_res("ERKEKLER - DOĞAL", res['M_Natural'])
    print_res("ERKEKLER - MANİPÜLE", res['M_Guided'])
    
    print("="*70)
def get_args_parser():
    parser = argparse.ArgumentParser(description="D3Rec", add_help=True)
    # inference.py içindeki parser kısmına ekle:
    # get_args_parser içindeki ilgili satırı bul ve değiştir:
    parser.add_argument('--str_cols', type=list, default=['UserId', 'MovieId', 'Rating', 'Timestamp', 'cate', 'user_pref'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--topK', default=[10, 20], type=int, nargs='+')
    parser.add_argument('--dataset_name', default='ml-1m', type=str)
    parser.add_argument('--split_ratio', default=[6, 2, 2], type=int, nargs="+")
    parser.add_argument('--drop_num', default=5, type=int)
    parser.add_argument('--dims', type=int, default=[600, 200], nargs="+")
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--dim_step', default=10, type=int)
    parser.add_argument('--beta_start', default=0.0001, type=float)
    parser.add_argument('--beta_end', default=0.02, type=float)
    parser.add_argument('--noise_scale', default=0.1, type=float)
    parser.add_argument('--steps', default=100, type=int)
    parser.add_argument('--sampling_steps', default=0, type=int)
    parser.add_argument('--noise_schedule', default="linear-var", type=str)
    parser.add_argument('--guide_w', default=5.0, type=float)
    
    # EKSİK OLAN PARAMETRELER BURAYA EKLENDİ (load_data için):
    parser.add_argument('--test_w_valid', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--snr', action='store_true')
    parser.add_argument('--sampling_noise', action='store_true')
    parser.add_argument('--drop_div', default=0.1, type=float)
    parser.add_argument('--lamda', default=1, type=float)
    parser.add_argument('--w_max', default=1, type=float)
    parser.add_argument('--w_min', default=0.2, type=float)

    return parser



if __name__ == '__main__':
    args = get_args_parser().parse_args()
    set_random_seed(random_seed=args.seed)
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    # get_paths fonksiyonu utils içinden gelir.
    # dataset_dir_path: .../dataset/ml-1m sonucunu vermeli
    dataset_dir_path, best_model_path = get_paths(args)
    
    # Preprocessing'den gender_dict fonksiyonunu çek
    from preprocessing import get_user_gender_dict
    # dataset_dir_path'i doğru veriyoruz (ml-1m klasörü)
    user_gender_map = get_user_gender_dict(dataset_dir_path)

    main(args, dataset_dir_path, best_model_path, user_gender_map)