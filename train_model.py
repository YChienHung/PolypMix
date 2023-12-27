from types import GeneratorType
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from tqdm import tqdm

import datasets
from utils.metrics import evaluate
from option import opt, method_model_dict, update_dataset
from utils.common import generate_model, TwoStreamBatchSampler, set_seed
from utils.loss import BceDiceLoss, entropy_loss
from utils.metrics import Metrics
from utils import ramps
import warnings


def get_current_consistency_weight(e, consistency, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(e, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def valid(model, valid_dataloader, total_batch, use_gpu):
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'Dice', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(valid_dataloader), total=total_batch):
            img, gt = data['image'], data['label']

            if use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            metrics.update(**evaluate(output, gt))

    metrics_result = metrics.mean()

    return metrics_result


def train(option):
    # set random seed
    set_seed(option.expID)

    # cuda device
    if option.use_gpu:
        torch.cuda.set_device(option.gpu_id)

    # load data
    train_data = getattr(datasets, option.dataset)(option.root, option.train_data_dir, mode='train')

    total_num = len(train_data)

    if option.label_mode == 'percentage':
        labeled_num = round(total_num * option.labeled_perc / 100)
    else:
        labeled_num = option.labeled_num

    print(
        f"Total training images: {total_num}, "
        f"labelled images: {labeled_num}, "
        f"labelled/total: {labeled_num / total_num * 100:.2f}%"
    )

    batch_sampler = TwoStreamBatchSampler(
        total_num, labeled_num, option.labeled_bs, option.batch_size - option.labeled_bs, shuffle=option.shuffle
    )
    train_dataloader = DataLoader(train_data, batch_sampler=batch_sampler, shuffle=False,
                                  num_workers=option.num_workers)

    valid_data = getattr(datasets, option.dataset)(option.root, option.valid_data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=option.num_workers)
    val_total_batch = len(valid_data)

    option.model = method_model_dict[option.method]
    model = generate_model(option)
    ema_model = generate_model(option, ema=True)
    model.train()

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=option.lr, momentum=option.mt, weight_decay=option.weight_decay)

    scheduler = LambdaLR(optimizer, lambda e: 1.0 - pow((e / option.nEpoch), option.power))

    kl_distance = None
    if option.use_gpu:
        criterion = BceDiceLoss().cuda()
        if option.method == 'URPC':
            kl_distance = nn.KLDivLoss(reduction='none').cuda()
    else:
        criterion = BceDiceLoss()
        if option.method == 'URPC':
            kl_distance = nn.KLDivLoss(reduction='none')

    # train
    print('Start training')
    print('---------------------------------\n')

    os.makedirs(
        os.path.join(
            option.checkpoints,
            f"{option.dataset}_{option.suffix}",
            f'{option.method}_{option.model}',
            f'exp{option.expID}'
        ),
        exist_ok=True
    )

    train_param_path = os.path.join(
        option.checkpoints,
        f"{option.dataset}_{option.suffix}",
        f'{option.method}_{option.model}',
        f'exp{option.expID}',
        'train_param.log'
    )

    with open(train_param_path, 'w') as f:
        f.write('\n'.join([f'{k}:{v}' for k, v in vars(option).items()]))
        f.close()

    iter_num = 0
    best_performance = 0.0
    best_epoch = 1
    best_metrics = {}

    # time
    epoch_time = list()
    batch_time = list()

    for epoch in range(option.nEpoch):
        start_epoch_time = time.time_ns()

        print(f"{'-' * 20} Epoch: {epoch + 1}, Method: {option.method}, Dataset: {option.dataset} {'-' * 20}")
        model.train()
        total_batch = int(labeled_num / option.labeled_bs)
        progress_bar = tqdm(enumerate(train_dataloader), total=total_batch)

        for batch_idx, data in progress_bar:

            start_batch_time = time.time_ns()

            images, gts = data['image'], data['label']
            labeled_img = images[:option.labeled_bs]
            unlabeled_img = images[option.labeled_bs:]

            if option.use_gpu:
                images = images.cuda()
                gts = gts.cuda()
                labeled_img = labeled_img.cuda()
                unlabeled_img = unlabeled_img.cuda()

            loss = None
            if option.method == 'MT':
                pass

            elif option.method == 'UA-MT':
                pass

            elif option.method == 'ICT':
                pass

            elif option.method == 'ADVENT':
                pass

            elif option.method == 'ClassMix':
                pass

            elif option.method == 'URPC':
                pass

            elif option.method == 'PolypMix':

                feature = torch.nn.functional.interpolate(
                    input=gts,
                    scale_factor=0.5 ** option.feat_level,
                    mode='bilinear',
                    align_corners=True,
                )

                unlabeled_group0, unlabeled_group1 = torch.chunk(unlabeled_img, 2, 0)

                with torch.no_grad():
                    ema_pred_gt0, ema_feature0 = ema_model(unlabeled_group0)
                    ema_pred_gt1, ema_feature1 = ema_model(unlabeled_group1)

                mix_factor_feature = torch.div(ema_feature1, ema_feature0 + ema_feature1)
                feature_mixed = torch.add(
                    torch.mul(ema_feature0, 1.0 - mix_factor_feature),
                    torch.mul(ema_feature1, mix_factor_feature),
                )

                feature_upsample0 = torch.nn.functional.interpolate(
                    ema_feature0, scale_factor=2 ** option.feat_level, mode='bilinear', align_corners=True
                )
                feature_upsample1 = torch.nn.functional.interpolate(
                    ema_feature1, scale_factor=2 ** option.feat_level, mode='bilinear', align_corners=True
                )

                mix_factor_image = torch.div(feature_upsample1, feature_upsample1 + feature_upsample0)
                batch_image_mixed = torch.add(
                    torch.mul(unlabeled_group0, 1.0 - mix_factor_image),
                    torch.mul(unlabeled_group1, mix_factor_image),
                )
                batch_gt_mixed = torch.add(
                    torch.mul(ema_pred_gt0, 1.0 - mix_factor_image),
                    torch.mul(ema_pred_gt1, mix_factor_image),
                )

                domain_mask0 = torch.mul(ema_pred_gt0 > ema_pred_gt1, ema_pred_gt0 > option.thresh)
                domain_mask1 = torch.mul(ema_pred_gt1 > ema_pred_gt0, ema_pred_gt1 > option.thresh)

                batch_gt_pseudo = torch.add(
                    torch.mul(ema_pred_gt0, domain_mask0) + torch.mul(ema_pred_gt1, domain_mask1),
                    torch.mul(batch_gt_mixed, torch.logical_not(torch.logical_or(domain_mask0, domain_mask1))),
                )

                input_batch = torch.cat([labeled_img, batch_image_mixed], dim=0)
                pred_gt, pred_feature = model(input_batch)

                # calc loss
                loss_supervised_gt = criterion(pred_gt[:option.labeled_bs], gts[:option.labeled_bs])
                loss_supervised_feature = criterion(pred_feature[:option.labeled_bs], feature[:option.labeled_bs])
                loss_supervised = loss_supervised_gt + loss_supervised_feature

                consistency_weight = get_current_consistency_weight(epoch + 1, option.consistency,
                                                                    option.consistency_rampup)
                loss_consistency_gt = criterion(pred_gt[option.labeled_bs:], batch_gt_pseudo)
                loss_consistency_feature = torch.mean((pred_feature[option.labeled_bs:] - feature_mixed) ** 2)
                loss = loss_supervised + consistency_weight * (loss_consistency_gt + loss_consistency_feature)

            elif option.method == 'Supervised_fully':
                pass

            elif option.method == 'Supervised':
                pass

            elif option.method == 'Supervised-Pra':
                pass

            elif option.method == 'Supervised-HRE':
                pass

            elif option.method == 'Supervised-HarD':
                pass

            else:
                print(f"error method: {option.method}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_batch_time = time.time_ns()
            batch_time.append((end_batch_time - start_batch_time) * 1e-9)

            update_ema_variables(model, ema_model, option.ema_decay, iter_num)
            iter_num = iter_num + 1
            progress_bar.set_postfix_str('loss: %.5s' % loss.item())

        end_epoch_time = time.time_ns()

        epoch_time.append((end_epoch_time - start_epoch_time) * 1e-9)

        scheduler.step()

        metrics_result = valid(model, valid_dataloader, val_total_batch, option.use_gpu)

        print(
            f"{'Valid':^10s}  : {', '.join([f'{k}:{v * 100:.2f}%' for k, v in metrics_result.items()])}"
        )
        performance = sum([v for k, v in metrics_result.items()])

        if (epoch + 1) % option.ckpt_period == 0:
            pth_path = os.path.join(
                option.checkpoints,
                f"{option.dataset}_{option.suffix}",
                f"{option.method}_{option.model}",
                f"exp{option.expID}",
                f"checkpoint_{epoch + 1}.pth"
            )

            torch.save(model.state_dict(), pth_path)
            print(
                f"best epoch:{best_epoch:^4d} - {', '.join([f'{k}:{v * 100:.2f}%' for k, v in best_metrics.items()])}"
            )

        if performance > best_performance:
            best_performance = performance
            best_metrics.update(metrics_result)
            best_epoch = epoch + 1

            pth_path = os.path.join(
                option.checkpoints,
                f"{option.dataset}_{option.suffix}",
                f'{option.method}_{option.model}',
                f"exp{option.expID}",
                f'checkpoint_best.pth'
            )
            torch.save(model.state_dict(), pth_path)

            print(
                f"best epoch:{best_epoch:^4d} - {', '.join([f'{k}:{v * 100:.2f}%' for k, v in best_metrics.items()])}"
            )

    log_path = os.path.join(
        option.checkpoints,
        f"{option.dataset}_{option.suffix}",
        f'{option.method}_{option.model}',
        f"exp{option.expID}",
        f'train.log'
    )

    with open(log_path, 'w') as f:
        f.write(f"best epoch:{best_epoch:>4d} - {', '.join([f'{k}:{v * 100:.2f}%' for k, v in best_metrics.items()])}")
        f.close()

    time_path = os.path.join(
        option.checkpoints,
        f"{option.dataset}_{option.suffix}",
        f'{option.method}_{option.model}',
        f"exp{option.expID}",
        f'time.log'
    )

    with open(time_path, 'w') as f:
        f.write(
            f"batch mean time:"
            f"{np.mean(batch_time):.6f}s - "
            f"{np.mean(batch_time) / 60:.6f}min - "
            f"{np.mean(batch_time) / 3600:.6f}h"
        )
        f.write('\n')

        f.write(
            f"epoch mean time:"
            f"{np.mean(epoch_time):.6f}s - "
            f"{np.mean(epoch_time) / 60:.6f}min - "
            f"{np.mean(epoch_time) / 3600:.6f}h"
        )
        f.write('\n')

        f.write(
            f"total time:"
            f"{np.sum(epoch_time) :.6f}s - "
            f"{np.sum(epoch_time) / 60:.6f}min - "
            f"{np.sum(epoch_time) / 3600:.6f}h"
        )
        f.write('\n')

        f.write(
            f"batch time(s):{', '.join([f'{t:.6f}' for t in batch_time])}"
        )
        f.write('\n')

        f.write(
            f"epoch time(s):{', '.join([f'{t:.6f}' for t in batch_time])}"
        )
        f.write('\n')

        f.write(
            f"epoch time(min):{', '.join([f'{t / 60:.6f}' for t in batch_time])}"
        )
        f.write('\n')

        f.write(
            f"epoch time(h):{', '.join([f'{t / 3600:.6f}' for t in batch_time])}"
        )
        f.write('\n')

        f.close()


if __name__ == '__main__':
    # close warning
    warnings.filterwarnings("ignore")

    print('--- PolpySeg Train ---')

    for method in method_model_dict.keys():
        opt.method = method

        train(opt)

    print('Done')
