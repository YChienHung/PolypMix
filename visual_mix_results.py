import numpy as np
import torch
from tqdm import tqdm
from option import opt, method_model_dict, update_dataset
import datasets
from torch.utils.data import DataLoader
from utils.common import generate_model
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms.functional as F

def visual(option):
    save_path = os.path.join(
        option.checkpoints,
        f"{option.dataset}_{option.suffix}",
        '005_dualMix',
        f"exp{option.expID}"
    )
    os.makedirs(save_path, exist_ok=True)

    option.method = 'PolypMix'
    option.model = method_model_dict[option.method]
    polyp_mix_model = generate_model(option)
    polyp_mix_model.eval()

    normal_list = os.listdir(os.path.join(option.image_path, 'normal'))
    tumor_list = os.listdir(os.path.join(option.image_path, 'tumor'))

    with torch.no_grad():
        bar = tqdm(enumerate(normal_list), total=len(normal_list))
        for idx, img_name in bar:
            normal_img = Image.open(os.path.join(option.image_path, 'normal', img_name)).convert('RGB')
            tumor_img = Image.open(os.path.join(option.image_path, 'tumor', tumor_list[idx])).convert('RGB')

            normal_img = F.resize(normal_img, [320, 320])
            normal_img = F.to_tensor(normal_img)

            tumor_img = F.resize(tumor_img, [320, 320])
            tumor_img = F.to_tensor(tumor_img)

            images = torch.stack([normal_img, tumor_img], dim=0)

            if option.use_gpu:
                images = images.cuda()

            input_image0, input_image1 = torch.chunk(images, 2, dim=0)

            # "PolypMix"
            with torch.no_grad():
                pred_gt0, feature0 = polyp_mix_model(input_image0)
                pred_gt1, feature1 = polyp_mix_model(input_image1)

            feature_upsample0 = torch.nn.functional.interpolate(
                feature0, scale_factor=2 ** option.feat_level, mode='bilinear', align_corners=True
            )
            feature_upsample1 = torch.nn.functional.interpolate(
                feature1, scale_factor=2 ** option.feat_level, mode='bilinear', align_corners=True
            )

            mix_factor = torch.div(feature_upsample1, feature_upsample1 + feature_upsample0)
            mix_polypmix = torch.mul(input_image0, 1.0 - mix_factor) + torch.mul(input_image1, mix_factor)

            plt.cla()
            plt.imshow(mix_polypmix.cpu().squeeze(0).permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f"{idx:03d}_{6:02d}_PolypMix.png"))


if __name__ == '__main__':
    print('--- PolypSeg Test---')

    visual(opt)

    print('Done')