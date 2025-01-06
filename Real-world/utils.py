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


def evaluate(args, model, diffusion, loader, gt_items, consumed_items, topK, item_category, n_cate, temperature=1, is_best=False):
    model.eval()
    num_user = gt_items.shape[0]
    user_idx_list = list(range(gt_items.shape[0]))

    predict_items = []
    target_items = []

    for user_id in range(num_user):
        target_items.append(gt_items[user_id, :].nonzero()[1].tolist())

    with torch.no_grad():
        for batch_idx, (x_0, prob, prob_pred) in enumerate(loader):
            start_batch_user_id = batch_idx * args.batch_size
            end_batch_user_id = start_batch_user_id + len(x_0)
            batch_consumed_items = consumed_items[user_idx_list[start_batch_user_id:end_batch_user_id]]

            x_0 = x_0.to(args.device)
            if temperature != 1:
                prob = adjust_div(prob, temperature)
            prob = prob.to(args.device)
            x_0_hat = diffusion.sample_new_interaction(model, x_0, prob, args.guide_w,
                                                       sampling_steps=args.sampling_steps,
                                                       sampling_noise=args.sampling_noise)

            x_0_hat[batch_consumed_items.nonzero()] = -np.inf
            _, indices = torch.topk(x_0_hat, topK[-1])
            indices = indices.detach().cpu().numpy().tolist()
            predict_items.extend(indices)

        if is_best is True:
            return compute_metric(target_items, predict_items, topK, item_category, n_cate)
        else:
            return compute_recall(target_items, predict_items, topk=20)


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