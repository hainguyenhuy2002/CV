from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
from utils import get_scale_from_dataset, resize_image

class EvalDataset(Dataset):
    def __init__(self, dataset, transform = None):
        super(EvalDataset, self).__init__()
        self.dataset = dataset
        self.scale = get_scale_from_dataset(dataset)
        self.transform = transform

    def __getitem__(self, idx):
        lr_image = Image.open(self.dataset[idx]['lr']).convert('RGB')
        hr_image = resize_image(lr_image, Image.open(self.dataset[idx]['hr']).convert('RGB'), scale=self.scale)
        lr = np.array(lr_image)
        hr = np.array(hr_image)
        lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255
        hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255
        if self.transform:
            lr, hr = self.transform(lr, hr)
            # label = self.transform(label)

        return lr, hr

    def __len__(self):
        return len(self.dataset)
    
    
class TrainDataset(Dataset):
    def __init__(self, dataset, transform = None, patch_size = 64):
        super(TrainDataset, self).__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.scale = get_scale_from_dataset(dataset)
        self.transform = transform
    
    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        lr_image = Image.open(self.dataset[idx]['lr']).convert('RGB')
        hr_image = resize_image(lr_image, Image.open(self.dataset[idx]['hr']).convert('RGB'), scale=self.scale)
        lr = np.array(lr_image)
        hr = np.array(hr_image)
        lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
        lr, hr = self.random_horizontal_flip(lr, hr)
        lr, hr = self.random_vertical_flip(lr, hr)
        lr, hr = self.random_rotate_90(lr, hr)
        lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255
        hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255
        
        if self.transform:
            lr, hr = self.transform(lr, hr)

        return lr, hr

    def __len__(self):
        return len(self.dataset)