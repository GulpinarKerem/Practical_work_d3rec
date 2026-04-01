import os
import torch
import random
import numpy as np


def set_random_seed(random_seed):
    torch.manual_seed(random_seed) # cpu
    torch.cuda.manual_seed(random_seed) # gpu
    np.random.seed(random_seed) # numpy
    random.seed(random_seed) # random and transforms
    torch.backends.cudnn.deterministic=True # cudnn


def adjust_div(prob, temperature):
    new_prob = prob + 1e-7  # prevent -inf of log
    logits = torch.log(new_prob) / temperature
    exp_logits = torch.exp(logits)
    new_prob = exp_logits / torch.sum(exp_logits, dim=-1)[:, None]

    return new_prob


def compute_recall(target_items, predict_items, topk):
    num_users = len(predict_items)
    sum_recall = 0.0

    for user_id in range(num_users):
        if len(target_items[user_id]) == 0:
            continue
        num_hit = 0

        for rank_idx in range(topk):
            if predict_items[user_id][rank_idx] in target_items[user_id]:
                num_hit += 1
        sum_recall += num_hit / len(target_items[user_id])

    recall = sum_recall / num_users
    return recall


def compute_metric(target_items, predict_items, topK, item_category, n_cate):
    precisions = []
    hit_ratios = []
    recalls = []
    ndcgs = []
    mrrs = []
    entropies = []
    coverages = []

    num_users = len(predict_items)

    for idx, k in enumerate(topK):
        sum_hitratio = sum_precision = sum_recall = sum_ndcg = sum_mrr = sum_entropy = sum_cov = 0.0
        for user_id in range(num_users):
            if len(target_items[user_id]) == 0:
                continue
            mrr_flag = True
            num_hit = user_mrr = dcg = 0

            for rank_idx in range(k):
                if predict_items[user_id][rank_idx] in target_items[user_id]:
                    num_hit += 1
                    dcg += 1.0 / np.log2(rank_idx + 2)
                    if mrr_flag:
                        user_mrr = 1.0 / (rank_idx + 1.0)
                        mrr_flag = False

            idcg = 0.0
            for rank_idx in range(min(len(target_items[user_id]), k)):
                idcg += 1.0 / np.log2(rank_idx + 2)
            user_ndcg = (dcg / idcg)

            cate_list = np.array([0.] * n_cate)
            for item in predict_items[user_id][:k]:
                cate_list[item_category[item]] += (1 / len(item_category[item]))

            sum_precision += num_hit / k
            sum_hitratio += (1 if num_hit > 0 else 0)
            sum_recall += num_hit / len(target_items[user_id])
            sum_ndcg += user_ndcg
            sum_mrr += user_mrr
            sum_cov += (np.count_nonzero(cate_list) / n_cate)

            etp = calculate_entropy(cate_list)
            sum_entropy += (etp / np.log2(n_cate))

        precision = sum_precision / num_users
        hit_ratio = sum_hitratio / num_users
        recall = sum_recall / num_users
        ndcg = sum_ndcg / num_users
        mrr = sum_mrr / num_users
        entropy = sum_entropy / num_users
        coverage = sum_cov / num_users

        precisions.append(precision)
        hit_ratios.append(hit_ratio)
        recalls.append(recall)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        entropies.append(entropy)
        coverages.append(coverage)

    return precisions, hit_ratios, recalls, ndcgs, mrrs, entropies, coverages


def evaluate(args, model, diffusion, loader, sp_test, sp_train_valid, topk, item_category, n_cate, temperature, user_gender_map, is_best=False):
    model.eval()
    
  
    
    results_all = {'target': [], 'pred': []}
    results_female = {'target': [], 'pred': []}
    results_male = {'target': [], 'pred': []}
    
    all_targets = []
    for u in range(sp_test.shape[0]):
        all_targets.append(sp_test.getrow(u).indices.tolist())

    current_user_idx = 0 

    with torch.no_grad():
        for x_0, prob, prob_pred in loader:
            x_0, prob = x_0.to(args.device), prob.to(args.device)
            modified_prob = prob.clone()
            
            for i in range(len(modified_prob)):
                uid_alt1 = current_user_idx + i + 1
                uid_alt2 = current_user_idx + i
                
                gender = user_gender_map.get(uid_alt1, user_gender_map.get(uid_alt2, "Unknown"))
                if gender == "Unknown":
                    gender = user_gender_map.get(str(uid_alt1), user_gender_map.get(str(uid_alt2), "M"))

                if gender == 'F':
                    modified_prob[i] = torch.tensor([0.1, 0.1, 0.8], device=args.device)

            modified_prob = adjust_div(modified_prob, temperature)
            x_0_hat = diffusion.sample_new_interaction(model, x_0, modified_prob, args.guide_w, args.sampling_steps)
            
            #mask n guess
            x_0_hat[x_0 > 0] = -np.inf
            _, indices = torch.topk(x_0_hat, k=max(topk), dim=-1)
            preds = indices.cpu().numpy().tolist()

            #groupem
            for i in range(len(preds)):
                uid_check = current_user_idx + i + 1
                gender_check = user_gender_map.get(uid_check, user_gender_map.get(current_user_idx + i, "M"))
                
                target = all_targets[current_user_idx + i]
                
                results_all['target'].append(target)
                results_all['pred'].append(preds[i])
                
                if gender_check == 'F':
                    results_female['target'].append(target)
                    results_female['pred'].append(preds[i])
                else:
                    results_male['target'].append(target)
                    results_male['pred'].append(preds[i])

            current_user_idx += len(x_0)

    #report
    print(f"\n" + "="*30 + f" REPORT (Temp: {temperature}) " + "="*30)
    
    def get_metrics(res):
        if not res['target']: return None
        return compute_metric(res['target'], res['pred'], [10, 20], item_category, n_cate)

    metrics_all = get_metrics(results_all)
    metrics_f = get_metrics(results_female)
    metrics_m = get_metrics(results_male)

    print("\n--- ALL USERS ---")
    print_metric_results([10, 20], metrics_all)
    
    print("\n--- FEMALE USERS (Modified/Harnessed) ---")
    if metrics_f: print_metric_results([10, 20], metrics_f)
    
    print("\n--- MALE USERS (Original/Natural) ---")
    if metrics_m: print_metric_results([10, 20], metrics_m)
    
    print("="*80)
    
    return metrics_all 


def calculate_entropy(cnt_cate_list):
    prob = cnt_cate_list / cnt_cate_list.sum()

    prob_pos = prob + 1e-7  # prevent -inf of log
    prob_pos = prob_pos / prob_pos.sum()
    entropy = -np.sum(prob_pos * np.log2(prob_pos))

    return entropy


def print_metric_results(topK, results):
    metric_list = ['Precision', 'Hit', 'Recall', 'nDCG', 'MRR', 'Entropy', 'Coverage']
    for k_idx, k in enumerate(topK):
        str_result = ''
        for idx, metric in enumerate(metric_list):
            str_metric = f'{metric}@{k:<5}'
            str_result += f'    {str_metric}: {results[idx][k_idx]:.4f}'
        print(str_result)


def make_directory(path):
    if os.path.exists(path) is False: os.makedirs(path)


def get_paths(args):
    dataset_dir_path = os.path.join(os.getcwd(), 'dataset', args.dataset_name)
    log_path = os.path.join(os.path.dirname(__file__), f'Best_models-C{args.drop_num}-{args.split_ratio}'.replace(" ", ""))
    log_dir_path = os.path.join(log_path, f'{args.dataset_name}')

    if args.save_model is True:
        make_directory(log_path)
        make_directory(log_dir_path)
    best_model_file_path = os.path.join(log_dir_path, 'best_model.pt')

    return (
        dataset_dir_path,
        best_model_file_path
    )