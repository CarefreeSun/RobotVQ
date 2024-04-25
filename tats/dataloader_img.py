import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import argparse

class ImageVideoDataset(Dataset):
    '''
    A dataset that batchify images into videos
    In the root directory contains files in the format: prefix_{scene_id}_{frame_id}_{view_id}.png
    we batchify the images with the same scene_id and view_id into a video clip with specified length
    each time when calling __getitem__, we randomly sample a video clip from the dataset
    '''
    def __init__(self, args, split='train', transform=None):
        self.root = args.dataroot
        self.transform = transform
        self.length = args.sequence_length
        self.split = split
        self.filenames = {}

        with open(os.path.join(self.root, f'{split}.txt'), 'r') as f:
            img_filenames = f.readlines()
        img_filenames = [img_filename.strip() for img_filename in img_filenames]

        for img_filename in img_filenames:
            _, scene_id, frame_id, view_id = img_filename.split('/')[-1].split('.')[0].split('_')
            key = (scene_id, view_id)
            if key not in self.filenames:
                self.filenames[key] = []
            self.filenames[key].append(img_filename)

        self.keys = list(self.filenames.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]
        filenames = self.filenames[key]
        if len(filenames) < self.length:
            raise ValueError('Not enough frames for the video clip')
        start = torch.randint(0, len(filenames) - self.length + 1, (1,)).item()
        video = []
        for i in range(start, start + self.length):
            img_filename = filenames[i]
            img = Image.open(os.path.join(self.root, img_filename))
            if self.transform:
                img = self.transform(img)
            video.append(img)
        video = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
        return {'video': video, 'key': key}

def get_image_video_dataloader(args, split='train'):
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageVideoDataset(args, split=split, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             num_workers=args.num_workers,
                                             shuffle=True if split == 'train' else False)
    return dataloader

        

