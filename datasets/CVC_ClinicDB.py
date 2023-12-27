import os
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tiff

from .kvasir_SEG import kvasir_SEG


class CVC_ClinicDB(kvasir_SEG):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(CVC_ClinicDB, self).__init__(root, data2_dir, mode, transform)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.gt_list[index]

        img = Image.fromarray(tiff.imread(img_path)).convert('RGB')
        gt = Image.fromarray(tiff.imread(gt_path)).convert('L')

        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)

        return {'id': self.id_list[index], 'image': data['image'], 'label': data['label']}


class CVC_ClinicDB_all_test(CVC_ClinicDB):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(CVC_ClinicDB_all_test, self).__init__(root, data2_dir, mode, transform)
