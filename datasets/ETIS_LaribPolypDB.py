import os
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tiff

from .kvasir_SEG import kvasir_SEG


class ETIS_LaribPolypDB(kvasir_SEG):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(ETIS_LaribPolypDB, self).__init__(root, data2_dir, mode, transform)

        self.gt_list = [os.path.join(self.data_path, 'masks', f"p{img_id}") for img_id in self.images_list]

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.gt_list[index]

        img = Image.fromarray(tiff.imread(img_path)).convert('RGB')
        gt = Image.fromarray(tiff.imread(gt_path)).convert('L')

        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)

        return {'id': self.id_list[index], 'image': data['image'], 'label': data['label']}


class ETIS_LaribPolypDB_all_test(ETIS_LaribPolypDB):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(ETIS_LaribPolypDB_all_test, self).__init__(root, data2_dir, mode, transform)
