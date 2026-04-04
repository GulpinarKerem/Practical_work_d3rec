import os
import torch
import random
import numpy as np


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True


def adjust_div(prob, temperature):
    new_prob = prob + 1e-7
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
        sum_hitratio = 0.0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_mrr = 0.0
        sum_entropy = 0.0
        sum_cov = 0.0

        for user_id in range(num_users):
            if len(target_items[user_id]) == 0:
                continue

            mrr_flag = True
            num_hit = 0
            user_mrr = 0.0
            dcg = 0.0

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

            user_ndcg = dcg / idcg if idcg > 0 else 0.0

            cate_list = np.array([0.0] * n_cate)
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


def evaluate(
    args,
    model,
    diffusion,
    loader,
    sp_test,
    sp_train_valid,
    topk,
    item_category,
    n_cate,
    temperature=1.0,
    is_best=False,
):
    """
    Phase 6 training/validation evaluation:
    - Uses the profile coming from the loader directly
    - No gender-based manipulation
    - This is the clean natural-profile evaluation used for model selection
    """
    model.eval()

    target_items = []
    for u in range(sp_test.shape[0]):
        target_items.append(sp_test.getrow(u).indices.tolist())

    predict_items = []

    with torch.no_grad():
        for x_0, prob_in, _ in loader:
            x_0 = x_0.to(args.device)
            prob_in = prob_in.to(args.device)

            if temperature != 1.0:
                prob_in = adjust_div(prob_in, temperature)

            x_0_hat = diffusion.sample_new_interaction(
                model,
                x_0,
                prob_in,
                args.guide_w,
                args.sampling_steps,
                args.sampling_noise if hasattr(args, "sampling_noise") else False,
            )

            # remove already-consumed items from candidate list
            x_0_hat[x_0 > 0] = -np.inf

            _, indices = torch.topk(x_0_hat, k=max(topk), dim=-1)
            preds = indices.cpu().numpy().tolist()
            predict_items.extend(preds)

    results = compute_metric(target_items, predict_items, topk, item_category, n_cate)

    if is_best:
        return results
    return results


def calculate_entropy(cnt_cate_list):
    prob = cnt_cate_list / cnt_cate_list.sum()

    prob_pos = prob + 1e-7
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
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_paths(args):
    dataset_dir_path = os.path.join(os.getcwd(), 'dataset', args.dataset_name)
    log_path = os.path.join(os.path.dirname(__file__), f'Best_models-C{args.drop_num}-{args.split_ratio}'.replace(" ", ""))
    log_dir_path = os.path.join(log_path, f'{args.dataset_name}')

    if args.save_model is True:
        make_directory(log_path)
        make_directory(log_dir_path)

    best_model_file_path = os.path.join(log_dir_path, 'best_model.pt')

    return dataset_dir_path, best_model_file_path