import torch
from tqdm import tqdm
from option import opt
from option import method_model_dict, update_dataset
from utils.metrics import evaluate
import datasets
from torch.utils.data import DataLoader
from utils.common import generate_model
from utils.metrics import Metrics
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import copy
import time

import sys
import importlib

importlib.reload(sys)
from prettytable import PrettyTable


def calc_statistics(scores, baseline_scores):
    # Calculate mean and standard deviation
    mean = np.mean(scores)
    std = np.std(scores)
    std_err = stats.sem(scores)

    # Calculate 95% Confidence Interval
    ci_lower, ci_upper = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(len(scores)))
    ci_lower_t, ci_upper_t = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=std_err)

    # Conduct a t-test (or appropriate test)
    # For instance, comparing to a baseline
    t_statistic, p_value = stats.ttest_ind(scores, baseline_scores)

    return mean, std, p_value, ci_lower, ci_upper, ci_lower_t, ci_upper_t


def calc_mean_std(scores):
    # Calculate mean and standard deviation
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std


def test(option):
    # cuda device
    if option.use_gpu:
        torch.cuda.set_device(option.gpu_id)

    gt_save_path = os.path.join(
        option.checkpoints,
        f"{option.dataset}_{option.suffix}",
        '002_GroundTruth',
        f"exp{option.expID}"
    )
    os.makedirs(gt_save_path, exist_ok=True)

    eval_path = os.path.join(option.checkpoints, f"{option.dataset}_{option.suffix}", '003_Evaluation')
    os.makedirs(eval_path, exist_ok=True)

    log_file_path = os.path.join(eval_path, f'{option.dataset}_eval_v2_exp{option.expID}.txt')

    print('loading data......')
    test_data = getattr(datasets, option.dataset)(option.root, option.test_data_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=option.num_workers)
    total_batch = int(len(test_data) / 1)

    # metrics_logger initialization
    metrics = Metrics(
        [
            'recall', 'specificity', 'precision', 'Dice', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean',
        ]
    )

    metrics_table = PrettyTable(['Method'] + [f'{k} (%)' for k in metrics.metrics.keys()])

    if option.save_image:
        for idx, data in tqdm(enumerate(test_dataloader), total=total_batch):
            indices, img, gts = data['id'], data['image'], data['label']

            plt.cla()
            plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(os.path.join(gt_save_path, f"{idx:03d}_{0:02d}_image.png"))

            plt.cla()
            plt.imshow(gts.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(gt_save_path, f"{idx:03d}_{1:02d}_gt.png"))

    total_metrics = dict()
    total_inference_time = dict()
    for model_index, method_name in enumerate(method_model_dict.keys()):
        option.method = method_name
        option.model = method_model_dict[method_name]

        load_ckpt_path = os.path.join(
            option.checkpoints,
            f"{option.dataset}_{option.suffix}" if option.select_checkpoint is None else option.select_checkpoint,
            f'{option.method}_{option.model}',
            'exp' + str(option.expID),
            f'{option.load_ckpt}.pth'
        )

        if not os.path.isfile(load_ckpt_path):
            continue

        model = generate_model(option, ema=True)
        model.eval()

        # none meaning, fix a bug
        img = torch.randn((1, 3, 320, 320))
        if option.use_gpu:
            img = img.cuda()
        model(img)

        metrics.clean()
        inference_time = list()
        with torch.no_grad():

            for idx, data in tqdm(enumerate(test_dataloader), total=total_batch):

                indices, img, gts = data['id'], data['image'], data['label']

                if option.use_gpu:
                    img = img.cuda()
                    gts = gts.cuda()

                start_inference_time = time.time_ns()

                output = model(img)
                metrics.update(**evaluate(output, gts))

                if isinstance(output, (list, tuple)):
                    pred_gt = output[0]
                else:
                    pred_gt = output

                end_inference_time = time.time_ns()
                inference_time.append((end_inference_time - start_inference_time) * 1e-9)

                if option.save_image:
                    plt.cla()
                    plt.imshow(pred_gt.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
                    plt.axis('off')
                    plt.savefig(os.path.join(gt_save_path, f"{idx:03d}_{model_index + 2:02d}_{method_name}.png"))

        metrics_result = metrics.mean()
        metrics_table.add_row([method_name] + [f'{v * 100:.2f}' for k, v in metrics_result.items()])

        total_metrics[method_name] = copy.deepcopy(metrics.metrics)
        total_inference_time[method_name] = copy.deepcopy(inference_time)

    statistics_table_norm = PrettyTable([f'Method (gauss, {len(test_data)}, low-up-p)'] + list(metrics.metrics.keys()))
    statistics_table_t = PrettyTable([f'Method (t, {len(test_data)}), low-up-p'] + list(metrics.metrics.keys()))
    distribute_table_norm = PrettyTable(
        [f'Method (gauss, {len(test_data)}, mean-half)'] + [f'{k} (%)' for k in metrics.metrics.keys()]
    )
    distribute_table_t = PrettyTable(
        [f'Method (t, {len(test_data)}, mean-half)'] + [f'{k} (%)' for k in metrics.metrics.keys()]
    )

    if 'PolypMix' in total_metrics.keys():
        semi_method = [
            'MT', 'UA-MT', 'ICT', 'ADVENT', 'ClassMix', 'URPC', 'Supervised_fully', 'Supervised',
            'Supervised-Pra', 'Supervised-HRE', 'Supervised-HarD'
        ]
        for method_name in semi_method:
            statistics_result_norm = dict()
            statistics_result_t = dict()
            distribute_result_norm = dict()
            distribute_result_t = dict()

            if method_name not in total_metrics.keys():
                continue
            for metric in total_metrics[method_name].keys():
                mean, std, p_value, ci_low_norm, c_upper_norm, ci_low_t, c_upper_t = calc_statistics(
                    total_metrics[method_name][metric],
                    total_metrics['PolypMix'][metric]
                )

                statistics_result_norm[metric] = [p_value, ci_low_norm * 100, c_upper_norm * 100]
                statistics_result_t[metric] = [p_value, ci_low_t * 100, c_upper_t * 100]

                distribute_result_norm[metric] = [mean * 100, (c_upper_norm - ci_low_norm) / 2 * 100]
                distribute_result_t[metric] = [mean * 100, (c_upper_t - ci_low_t) / 2 * 100]

            statistics_table_norm.add_row(
                [method_name] + [f'{v[1]:.2f}%-{v[2]:.2f}% ({v[0]:.6f})' for k, v in statistics_result_norm.items()]
            )
            statistics_table_t.add_row(
                [method_name] + [f'{v[1]:.2f}%-{v[2]:.2f}% ({v[0]:.6f})' for k, v in statistics_result_t.items()]
            )

            distribute_table_norm.add_row(
                [method_name] + [f'{v[0]:.2f}-{v[1]:.2f}' for k, v in distribute_result_norm.items()]
            )
            distribute_table_t.add_row(
                [method_name] + [f'{v[0]:.2f}-{v[1]:.2f}' for k, v in distribute_result_t.items()]
            )

    mean_std_table = PrettyTable(
        [f'Method ({len(test_data)}, mean-std)'] + [f'{k} (%)' for k in metrics.metrics.keys()]
    )
    for method_name in total_metrics.keys():
        mean_std_result = dict()

        for metric in total_metrics[method_name].keys():
            mean, std = calc_mean_std(total_metrics[method_name][metric])
            mean_std_result[metric] = [mean * 100, std * 100]

        mean_std_table.add_row(
            [method_name] + [f"{v[0]:.2f}-{v[1]:.2f}" for k, v in mean_std_result.items()]
        )

    with open(log_file_path, 'w') as f:
        f.write(str(metrics_table))
        f.write('\n\n')
        f.write(str(statistics_table_norm))
        f.write('\n\n')
        f.write(str(distribute_table_norm))
        f.write('\n\n')
        f.write(str(statistics_table_t))
        f.write('\n\n')
        f.write(str(distribute_table_t))
        f.write('\n\n')

        f.write(str(mean_std_table))
        f.close()

    time_path = os.path.join(
        eval_path,
        f'inference_time_v2_exp{option.expID}.log'
    )

    with open(time_path, 'w') as f:
        max_name_length = max([len(name) for name in total_inference_time.keys()])

        table = PrettyTable(['type'] + list(total_inference_time.keys()))
        table.add_row(['inference time (s)'] + [f'{np.mean(v):.6f}' for k, v in total_inference_time.items()])
        table.add_row(['inference time (min)'] + [f'{np.mean(v) / 60:.6f}' for k, v in total_inference_time.items()])
        table.add_row(['inference time (h)'] + [f'{np.mean(v) / 3600:.6f}' for k, v in total_inference_time.items()])

        f.write(str(table))

        f.write('\n\n')
        f.write(
            '\n'.join(
                [
                    f"{k:^{max_name_length}s}:{', '.join([f'{val:2.6f}s' for val in v])}"
                    for k, v in total_inference_time.items()
                ]
            )
        )


if __name__ == '__main__':
    print('--- PolypSeg Test---')

    test(opt)

    print('Done')

# 直接测试
# python test_model.py --dataset CVC_300 --suffix number --expID 7777
# python test_model.py --dataset ETIS_LaribPolypDB_all_test --select_checkpoint kvasir_SEG_number --suffix kvasir_direct --expID 7777

# python test_model.py --dataset kvasir_SEG --suffix no_shuffle

# 训练指定数量标签的模型
# python test_model.py --dataset kvasir_SEG --suffix number --expID 9999

# 测试加载权重训练的模型
# python test_model.py --dataset ETIS_LaribPolypDB --suffix weight

# 训练指定比例标签的模型
# python test_model.py --dataset ETIS_LaribPolypDB --suffix percentage --expID 7777

# kvasir_SEG CVC_300 CVC_ClinicDB ETIS_LaribPolypDB


# python test_model.py --dataset CVC_300 --select_checkpoint kvasir_SEG_percentage --suffix direct
