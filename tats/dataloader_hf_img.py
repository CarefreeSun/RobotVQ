import torch
from datasets import Dataset
from torchvision import transforms
import os
from PIL import Image
import argparse
import json
import random

def gen(shards, args, transform):
    for shard in shards:
        with open(shard, "r") as f:
            for line in f:
                instance_info = json.loads(line)
                num_frames = instance_info["Frame_number"]
                ret = {}
                # randomly select num_input+num_output consecutive frames
                if num_frames < args.sequence_length:
                    continue
                start_frame = random.randint(0, num_frames - args.sequence_length)
                video = []
                for i in range(start_frame, start_frame + args.sequence_length):
                    img = Image.open(instance_info["Visual"][i]).convert('RGB')
                    img = transform(img)
                    video.append(img)
                video = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
                ret['video'] = video
                yield ret

def get_image_video_dataset(args, transform, split='train'):
    root = args.dataroot
    file_format = split + '_{}.json'
    shards = [os.path.join(root, split, file_format.format(i)) for i in range(len(os.listdir(os.path.join(root, split))))]
    ds = Dataset.from_generator(gen, gen_kwargs={"args": args, "shards": shards, "transform": transform})
    return ds

def get_image_video_dataloader(args, split='train'):
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_image_video_dataset(args, transform, split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             num_workers=args.num_workers,
                                             shuffle=True if split == 'train' else False)
    return dataloader

        

