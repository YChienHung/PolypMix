import os
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms

from .kvasir_SEG import kvasir_SEG


class CVC_300(kvasir_SEG):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(CVC_300, self).__init__(root, data2_dir, mode, transform)


class CVC_300_all_test(CVC_300):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(CVC_300_all_test, self).__init__(root, data2_dir, mode, transform)
