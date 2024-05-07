import json
import argparse
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGANDeepSpeed, VideoData, get_image_dataloader
from tats.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

parser = argparse.ArgumentParser()

parser.add_argument("--nodes", type=int, default=1, help="nodes")
parser.add_argument("--devices", type=int, default=8, help="e.g., gpu number")
parser.add_argument("--default_root_dir", type=str, default="debug")
parser.add_argument("--max_steps", type=int, default=2000000, help="max_steps")

# model args
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--n_codes', type=int, default=2048)
parser.add_argument('--n_hiddens', type=int, default=32)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--downsample', nargs='+', type=int, default=(1, 16, 16))
parser.add_argument('--disc_channels', type=int, default=64)
parser.add_argument('--disc_layers', type=int, default=3)
parser.add_argument('--discriminator_iter_start', type=int, default=50000)
parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
parser.add_argument('--image_gan_weight', type=float, default=1.0)
parser.add_argument('--video_gan_weight', type=float, default=1.0)
parser.add_argument('--l1_weight', type=float, default=4.0)
parser.add_argument('--gan_feat_weight', type=float, default=4.0)
parser.add_argument('--perceptual_weight', type=float, default=4.0)
parser.add_argument('--i3d_feat', action='store_true')
parser.add_argument('--restart_thres', type=float, default=1.0)
parser.add_argument('--no_random_restart', action='store_true')
parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'group'])
parser.add_argument('--padding_type', type=str, default='replicate', choices=['replicate', 'constant', 'reflect', 'circular'])

# data args
parser.add_argument(
    "--dataroot", type=str, default="/mnt/data-rundong/TATS/data_lists"
)
parser.add_argument("--sequence_length", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--src', type=str, default='mnt/robotdata/bridge2/bridge2_gpt4v_annotated_merged_output.jsonl')
parser.add_argument('--dst_dir', type=str, default='/mnt/azureml/cr/j/f5579abda37a4331a8394bb5cd966da5/exe/wd/bridge_gpt4v_tokenized')
parser.add_argument('--config_path', type=str, default='configs/vqgan_imagenet_f16_1024.yaml')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--num_shards', type=int, default=8)

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'

dst_path = os.path.join(args.dst_dir, f'data_bridge2_processed_{args.gpu_id}.jsonl')
os.makedirs(args.dst_dir, exist_ok=True)

model = VQGANDeepSpeed(args).eval().to(device)

# load jsonl file, each line contains ID, Frame_Number, Text, Visual, Action fileds
# tokenize the Images using the model, and tokenize the action using discrete tokenization ([-1, 1] -> [0,... 255])
with open(args.src, 'r') as f:
    lines = f.readlines()
    num_data = len(lines)
    shard_size = num_data // args.num_shards + 1
    for line in tqdm(lines[args.gpu_id * shard_size: args.gpu_id * shard_size + shard_size]):
        instance_info = json.loads(line)
        # each instance_info contains multiple frames, thus we need to handle them batch by batch
        num_frames = instance_info["Frame_number"]

        tokens = torch.zeros([num_frames, 256], dtype=torch.int32).to(device)

        with torch.no_grad():
            for j in range((num_frames - 1) // args.batch_size + 1):
                start = j * args.batch_size
                end = min((j + 1) * args.batch_size, num_frames)
                images = torch.zeros(end - start, 3, 256, 256)

                for i in range(start, end):
                    image_path = instance_info['Visual'][i]
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((256, 256))
                    image = np.array(image).transpose(2, 0, 1)
                    image = torch.tensor(image).float() / 255.0
                    images[i - start] = image

                images = images.to(device)
                _, _, info = model.encode(images)
                tokens[start:end] = info[-1].view(end - start, -1)

        new_json = {}
        for k, v in instance_info.items():
            if k not in ['Visual', 'Action']:
                new_json[k] = v
        new_json['Visual'] = tokens.cpu().numpy().tolist()
        action_raw = np.array(instance_info['Action'])
        action = (action_raw + 1) * 127.5
        new_json['Action'] = action.astype(np.int32).tolist()
        
        with open(dst_path, 'a') as f:
            f.write(json.dumps(new_json) + '\n')

# randomly sample 90% of the data for training, and 10% for validation
os.makedirs(os.path.join(args.dst_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(args.dst_dir, 'val'), exist_ok=True)

with open(dst_path, 'r') as f:
    lines = f.readlines()
    np.random.shuffle(lines)
    num_train = int(len(lines) * 0.9)
    train_lines = lines[:num_train]
    val_lines = lines[num_train:]
    with open(os.path.join(args.dst_dir, 'train', f'data_bridge2_processed_{args.gpu_id}.jsonl'), 'w') as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(args.dst_dir, 'val', f'data_bridge2_processed_{args.gpu_id}.jsonl'), 'w') as f:
        for line in val_lines:
            f.write(line)
