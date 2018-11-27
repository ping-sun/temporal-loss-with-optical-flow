import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataGenerator(Dataset):
    def __init__(self, data, phase='train'):
        if type(data) is list:
            self.image, self.label = [], []
            for d in data:
                self.image.extend(d.image)
                self.label.extend(d.label)
        else:
            self.image = data.image
            self.label = data.label

        def base_transform(t='img'):
            if t=='label':
                interpolation = Image.NEAREST
            else:
                interpolation = Image.BILINEAR
            return {
                'train': transforms.Compose([
                    transforms.Resize(640, interpolation=interpolation),
                    transforms.RandomRotation(15, resample=interpolation),
                    transforms.RandomResizedCrop(512, scale=(0.8, 1.5), ratio=(1.2, 0.85), interpolation=interpolation),
                    transforms.RandomHorizontalFlip(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((480, 640), interpolation=interpolation),
                ]),
                'test': transforms.Compose([
                    transforms.Resize((480, 640), interpolation=interpolation),
                ]),
            }[phase]
        img_transform = {
            'train': transforms.Compose([
                base_transform(),
                transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
            ]),
            'val': transforms.Compose([
                base_transform(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                base_transform(),
                transforms.ToTensor(),
            ]),
        }[phase]

        def image_transform(image, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            image = img_transform(image)
            if image.size(0)==1:
                image = image.repeat(3, 1, 1)
            return image

        self.image_transform = image_transform

        def label_transform(label, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            label = np.array(base_transform('label')(label))
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy((label>0).astype(np.int64))
        self.label_transform = label_transform

    def __getitem__(self, idx):

        rand_seed = np.random.randint(9223372036854775808)

        image = Image.open(self.image[idx]).convert("RGB")
        image = self.image_transform(image, rand_seed)

        label = Image.open(self.label[idx])
        label = self.label_transform(label, rand_seed)

        sample = {
            'image': image, 'label': label,
        }

        return sample

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':

    from imagereader import imagefile
    from torch.utils.data import DataLoader
    mhp = imagefile('dataset/LV-MHP-v2', 'list/train.txt')
    supervisely = imagefile('dataset/Supervisely', 'train.txt',
                            img_dir='SuperviselyImages', label_dir='SuperviselyMasks')
    data_dataset = DataGenerator([mhp, supervisely], phase='val')
    batch_size = 8
    num_workers = 8
    data_loader = DataLoader(
        data_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    import time
    t = time.time()
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(data_loader)):
        image, label = batch['image'], batch['label']
        # print(i, image.shape, label.shape)
    print(time.time() - t, 'seconds', 'batch_size', batch_size, 'num_workers', num_workers)
