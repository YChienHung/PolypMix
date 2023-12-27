import os
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms


# KavSir-SEG Dataset
class kvasir_SEG(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None, cache=False):
        super(kvasir_SEG, self).__init__()
        self.data_path = os.path.join(root, data2_dir)

        self.id_list = []
        self.img_list = []
        self.gt_list = []

        self.images_list = os.listdir(os.path.join(self.data_path, 'images'))
        self.images_list = sorted(self.images_list)
        for img_id in self.images_list:
            self.id_list.append(img_id.split('.')[0])
            self.img_list.append(os.path.join(self.data_path, 'images', img_id))
            self.gt_list.append(os.path.join(self.data_path, 'masks', img_id))

        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation(90),
                    RandomZoom((0.9, 1.1)),
                    # Translation(10),
                    RandomCrop((224, 224)),
                    ToTensor(),
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    ToTensor(),
                ])
        self.transform = transform

        self.cache = cache
        if self.cache:
            self.cache_img = list()
            self.cache_gt = list()
            for index in range(len(self.img_list)):
                img_path = self.img_list[index]
                gt_path = self.gt_list[index]

                self.cache_img[index] = Image.open(img_path).convert('RGB')
                self.cache_gt[index] = Image.open(gt_path).convert('L')

    def __getitem__(self, index):
        if self.cache:
            img = self.cache_img[index]
            gt = self.cache_gt[index]
        else:
            img_path = self.img_list[index]
            gt_path = self.gt_list[index]

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')

        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)

        return {'id': self.id_list[index], 'image': data['image'], 'label': data['label']}

    def __len__(self):
        return len(self.img_list)
