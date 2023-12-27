import argparse
import os
from torch import float16

method_model_dict = {
    # 'MT': 'UUNet',
    # 'UA-MT': 'UUNet',
    # 'ICT': 'UUNet',
    # 'ADVENT': 'UUNet',
    # 'ClassMix': 'UUNet',
    # 'URPC': 'URPC',
    'PolypMix': 'PolypUU',
    # 'Supervised_fully': 'UUNet',
    # 'Supervised': 'UUNet',
    # 'Supervised-Pra': 'PraNet',
    # 'Supervised-HRE': 'HreNet',
    # 'Supervised-HarD': 'HarDMSEG',
}

parse = argparse.ArgumentParser(description='PyTorch Polyp Segmentation')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/home/data/gastrointestinal/project/PolypMix/data')
# kvasir_SEG CVC_300 CVC_ClinicDB ETIS_LaribPolypDB
parse.add_argument('--dataset', type=str, default='ETIS_LaribPolypDB')
parse.add_argument('--train_data_dir', type=str, default='kvasir-SEG/Train')
parse.add_argument('--valid_data_dir', type=str, default='kvasir-SEG/Valid')
parse.add_argument('--test_data_dir', type=str, default='kvasir-SEG/Test')

"-------------------testing option-----------------------"
parse.add_argument('--select_checkpoint', type=str, default=None)
parse.add_argument('--save_image', type=bool, default=False)

"-------------------training option-----------------------"
parse.add_argument('--checkpoints', type=str, default='/home/data/gastrointestinal/project/PolypMix/checkpoints')
parse.add_argument('--pretrain', type=str, default=None, help='use pretrained model weight, default: None')
parse.add_argument('--suffix', type=str, default='normal')
parse.add_argument('--shuffle', type=str, default=True)

# method = "MT" "UA-MT" "ICT" "ADVENT" "ClassMix" "URPC" "PolypMix"
# "Supervised-fully" "Supervised" "Supervised-Pra" "Supervised-HRE" "Supervised-HarD"
parse.add_argument('--method', type=str, default='MT', help='Train Method')
parse.add_argument('--expID', type=int, default=8888)
parse.add_argument('--load_ckpt', type=str, default='checkpoint_best')
parse.add_argument('--nEpoch', type=int, default=300)
parse.add_argument('--batch_size', type=float, default=4)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--use_gpu', type=bool, default=True)
parse.add_argument('--gpu_id', type=int, default=0)
parse.add_argument('--model', type=str, default='UUNet')
parse.add_argument('--ckpt_period', type=int, default=10)

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--power', type=float, default=0.9)
parse.add_argument('--num_class', type=int, default=1)
parse.add_argument('--thresh', type=float, default=0.7, help='pseudo threshold')

"-------------------label and unlabeled-----------------------"
parse.add_argument('--label_mode', type=str, default='percentage', help='percentage or number')
parse.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size')
parse.add_argument('--labeled_perc', type=int, default=15, help='percentage of labeled')
parse.add_argument('--labeled_num', type=int, default=90, help='min number of labeled')

"-------------------costs-----------------------"
parse.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parse.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parse.add_argument('--consistency', type=float, default=0.1, help='consistency')
parse.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

"-------------------PolypMix-----------------------"
parse.add_argument('--feat', type=int, default=4, help='feature layer')
parse.add_argument('--loss', type=str, default='deep_up_out', help='deep supervision loss')

"-------------------PolypMix_feat-----------------------"
parse.add_argument('--feat_level', type=int, default=4, help='feature layer')
parse.add_argument('--consist', type=str, default='False', help='feature-level consistency')
parse.add_argument('--dropout', type=float, default=0.1, help='dropout in sideout block')

"-------------------Test Visualization-----------------------"
parse.add_argument('--image_path', type=str, default=None, help='test image path')

opt = parse.parse_args()


def update_dataset(dataset_name=None):
    if dataset_name is not None:
        opt.dataset = dataset_name
    opt.train_data_dir = opt.dataset + '/Train'
    opt.valid_data_dir = opt.dataset + '/Valid'
    opt.test_data_dir = opt.dataset + '/Test'


update_dataset()
