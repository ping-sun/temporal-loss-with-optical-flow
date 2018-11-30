import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import pickle
import gzip
import math

class DataGenerator(Dataset):
    def __init__(self, data, phase='train'):
        self.phase = phase
        if type(data) is list:
            self.image_list, self.label_list, self.flow_list = [], [], []
            self.background_image = []
            self.background_label = []
            for d in data:
                self.image_list.extend(d.image_list)
                self.label_list.extend(d.label_list)
                self.flow_list.extend(d.flow_list)
                self.background_image.extend(d.background_image)
                self.background_label.extend(d.background_label)
        else:
            self.image_list = data.image_list
            self.label_list = data.label_list
            self.flow_list  = data.flow_list

            self.background_image = data.background_image
            self.background_label = data.background_label

        def base_transform(t='img'):
            if t=='label':
                interpolation = Image.NEAREST
            else:
                interpolation = Image.BILINEAR
            return {
                'train': transforms.Compose([
                    # transforms.Resize(640, interpolation=interpolation),
                    # transforms.RandomRotation(15, resample=interpolation),
                    # transforms.RandomResizedCrop(512, scale=(0.8, 1.5), ratio=(1.2, 0.85), interpolation=interpolation),
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((512, 512))
                ]),
                'val': transforms.Compose([
                    # transforms.Resize((480, 640), interpolation=interpolation),
                    transforms.RandomCrop((512, 512))
                ]),
                'test': transforms.Compose([
                    # transforms.Resize((480, 640), interpolation=interpolation),
                    transforms.RandomCrop((512, 512))
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

        #LOAD IMAGE AND FLOW
        prev = cv2.imread(self.image_list[idx][0])
        cur = cv2.imread(self.image_list[idx][1])
        prev_label = cv2.imread(self.label_list[idx][0])
        label = cv2.imread(self.label_list[idx][1])
        flow1 = pickle.loads(gzip.GzipFile(self.flow_list[idx][0], 'rb').read())  #'forward_0_5.pkl'
        flow2 = pickle.loads(gzip.GzipFile(self.flow_list[idx][1], 'rb').read())  #'backward_5_0.pkl'
        h,w = prev.shape[:2]

        #print("read???:",prev.shape, cur.shape, label.shape, flow1.shape, flow2.shape)
        flow1 = flow1.astype(np.float32)
        flow2 = flow2.astype(np.float32)


        #Augment parameters
        resize_rate = 640.0 / min(h,w)
        if h < w:
            rh = 640
            rw = int(w * resize_rate)
        else:
            rw = 640
            rh = int(h * resize_rate)
        resize_shape = (rw, rh)
        #print(resize_shape)
        crop_size = 512
        hflip = random.randint(0,1)  #flip
        beta = random.randint(-15,15)  #rotate
        cos = math.cos(math.radians(beta))
        sin = math.sin(math.radians(beta))
        Rotation = cv2.getRotationMatrix2D((rw/2,rh/2), beta, 1) #1==resize ratio
        rotm = np.array(([[cos,-sin,(1-cos)*rw/2+rh/2*sin],[sin, cos,(1-cos)*rh/2-rw/2*sin],[0,0,1]]))
        rotate_shape = resize_shape


        #RESIZE
        #target.shape[0] = shape[0] * fy
        #target.shape[1] = shape[1] * fx
        prev = cv2.resize(prev, resize_shape, interpolation=cv2.INTER_LINEAR)
        cur  = cv2.resize(cur, resize_shape, interpolation=cv2.INTER_LINEAR)
        label  = cv2.resize(label, resize_shape, interpolation=cv2.INTER_LINEAR)
        prev_label  = cv2.resize(prev_label, resize_shape, interpolation=cv2.INTER_LINEAR)
        flow1 = cv2.resize(flow1, resize_shape, interpolation=cv2.INTER_LINEAR)
        flow2 = cv2.resize(flow2, resize_shape, interpolation=cv2.INTER_LINEAR)
        flow1 *= resize_rate
        flow2 *= resize_rate# rf1[:,:,1] *= fy rf1[:,:,0] *= fx

        #FLIP
        if hflip==1:
            prev  = cv2.flip(prev, 1)
            cur   = cv2.flip(cur, 1)
            label = cv2.flip(label, 1)
            prev_label = cv2.flip(prev_label, 1)
            flow1 = cv2.flip(flow1, 1)
            flow2 = cv2.flip(flow2, 1)
            flow1[:,:,0] *= -1
            flow2[:,:,0] *= -1


        #ROTATE
        prev  = cv2.warpAffine(prev, Rotation, rotate_shape)
        cur   = cv2.warpAffine(cur, Rotation, rotate_shape)
        label = cv2.warpAffine(label, Rotation, rotate_shape)
        prev_label = cv2.warpAffine(prev_label, Rotation, rotate_shape)
        flow1 = cv2.warpAffine(flow1, Rotation, rotate_shape)
        flow2 = cv2.warpAffine(flow2, Rotation, rotate_shape)

        rotshape = flow1.shape

        flow1 = flow1.reshape(-1,2)
        flow2 = flow2.reshape(-1,2)
        rot0 = np.ones(flow1.shape[0])[:,np.newaxis]
        flow1 =  np.concatenate((flow1, rot0),axis=1)
        flow2 =  np.concatenate((flow2, rot0),axis=1)

        flow1 = flow1.dot(rotm).reshape(rh,rw,3)[:,:,:2]
        flow2 = flow2.dot(rotm).reshape(rh,rw,3)[:,:,:2]

        #CROP
        crop_shape = prev.shape[:2] #3,H,W
        crop_h = random.randint(0,crop_shape[0]-crop_size)
        crop_w = random.randint(0,crop_shape[1]-crop_size)

        prev  = prev[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
        cur   = cur[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
        label = label[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
        prev_label = prev_label[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
        flow1 = flow1[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
        flow2 = flow2[crop_h:crop_h+crop_size, crop_w:crop_w+crop_size, :]
        
        #Transpose
        prev = prev.transpose(2,0,1) / 255.0
        cur = cur.transpose(2,0,1) / 255.0
        label = (label.transpose(2,0,1)>0).astype(np.int64)
        prev_label = (prev_label.transpose(2,0,1)>0).astype(np.int64)[0]
        flow1 = flow1.transpose(2,0,1).astype(np.int64)
        flow2 = flow2.transpose(2,0,1).astype(np.int64)
        
        #To Torch
        prev = torch.from_numpy(prev).float()
        cur =  torch.from_numpy(cur).float()
        label = torch.from_numpy(label)#.float()
        prev_label = torch.from_numpy(prev_label)#.float()
        flow1 = torch.from_numpy(flow1).float()
        flow2 = torch.from_numpy(flow2).float()
        #print("tensor shape:", prev.shape, cur.shape, label.shape, flow1.shape, flow2.shape)
        


        sample = {
            'prev':prev, 'cur':cur, 'prev_label':prev_label, 'label': label, 'flow1' : flow1, 'flow2' : flow2
        }

        return sample


    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    pass
