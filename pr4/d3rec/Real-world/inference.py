import os, time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset_load import load_data
from utils_train import *
from utils import *
from models import *

def main():
    args = get_args_parser().parse_args()
    set_random_seed(random_seed=args.seed)
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    
    dataset_dir_path, best_model_path = get_paths(args)
    from preprocessing import get_user_gender_dict
    user_gender_map = get_user_gender_dict(dataset_dir_path)

    print(f"Starting Inference. Device: {args.device}")

    
    dataset, _, _, test_dataset, matrix_F = load_data(args, dataset_dir_path)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    diffusion = Diffusion(steps=args.steps, beta_start=args.beta_start, beta_end=args.beta_end,
                          noise_scale=args.noise_scale, noise_schedule=args.noise_schedule, device=args.device)
    
    model = D3Rec(dims=args.dims, n_item=dataset.num_items, n_cate=dataset.num_cate,
                  dim_step=args.dim_step, dropout=args.dropout).to(args.device)

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint.state_dict())
        print(f"Model loaded correctly from {best_model_path}")
        
    
    model.eval()

    #scenarios
    scenarios = [
        ("UNIFORM (Balanced)", [0.333, 0.333, 0.334]),
        ("PURE HIGH-POP (Popular Only)", [1.0, 0.0, 0.0]),
        ("PURE MID-POP",[0.0,1.0,0.0]),
        ("PURE LOW-POP (Niche Only)", [0.0, 0.0, 1.0])
    ]

    for name, target_vec in scenarios:
        print(f"\n>>> RUNNING CASE: {name} with target {target_vec}")
        run_evaluation(args, model, diffusion, loader, dataset, user_gender_map, target_vec)

def run_evaluation(args, model, diffusion, loader, dataset, user_gender_map, target_prob_list):
    res = {'ALL': {'t': [], 'p': []}, 'F': {'t': [], 'p': []}, 'M': {'t': [], 'p': []}}
    
    
    all_targets = []
    for u in range(dataset.sp_test.shape[0]):
        all_targets.append(dataset.sp_test.getrow(u).indices.tolist())

    u_idx = 0
    with torch.no_grad():
        for x_0, _, _ in loader:
            x_0 = x_0.to(args.device)
            target_prob = torch.tensor(target_prob_list, device=args.device).repeat(x_0.shape[0], 1)
            x_0_gui = diffusion.sample_new_interaction(model, x_0, target_prob, args.guide_w, args.sampling_steps)
            x_0_gui[x_0 > 0] = -np.inf # İzlediklerini çıkar
            
            preds = torch.topk(x_0_gui, k=max(args.topK), dim=-1)[1].cpu().numpy().tolist()

            for i in range(len(x_0)):
                uid = u_idx + i + 1
                gender = user_gender_map.get(uid, "M")
                target = all_targets[u_idx + i]
                
                res['ALL']['t'].append(target); res['ALL']['p'].append(preds[i])
                if gender == 'F':
                    res['F']['t'].append(target); res['F']['p'].append(preds[i])
                else:
                    res['M']['t'].append(target); res['M']['p'].append(preds[i])
            u_idx += len(x_0)

    for key in ['ALL', 'FEMALE', 'MALE']:
        k = key[0] if key != 'ALL' else 'ALL'
        if not res[k]['t']: continue
        metrics = compute_metric(res[k]['t'], res[k]['p'], args.topK, dataset.item_category, dataset.num_cate)
        print(f"--- {key} Results ---")
        print_metric_results(args.topK, metrics)

def get_args_parser():
    parser = argparse.ArgumentParser(description="D3Rec Inference", add_help=True)
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
    
    # Dataset ve model yükleme için gerekli flagler
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
    main()