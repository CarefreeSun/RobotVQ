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
from tats import VQGANVisionAction, VideoData, get_image_action_dataloader, count_parameters
from tats.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from torchvision import transforms

parser = argparse.ArgumentParser()

parser.add_argument("--nodes", type=int, default=1, help="nodes")
parser.add_argument("--devices", type=int, default=8, help="e.g., gpu number")
parser.add_argument("--default_root_dir", type=str, default="debug")
parser.add_argument("--max_steps", type=int, default=2000000, help="max_steps")

# model args
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--n_codes', type=int, default=16384)
parser.add_argument('--n_hiddens', type=int, default=32)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--downsample', nargs='+', type=int, default=(2, 16, 16))
parser.add_argument('--disc_channels', type=int, default=64)
parser.add_argument('--disc_layers', type=int, default=3)
parser.add_argument('--discriminator_iter_start', type=int, default=50000)
parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
parser.add_argument('--image_gan_weight', type=float, default=1.0)
parser.add_argument('--video_gan_weight', type=float, default=1.0)
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--gan_feat_weight', type=float, default=4.0)
parser.add_argument('--perceptual_weight', type=float, default=4.0)
parser.add_argument('--i3d_feat', action='store_true')
parser.add_argument('--restart_thres', type=float, default=1.0)
parser.add_argument('--no_random_restart', action='store_true')
parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'group'])
parser.add_argument('--padding_type', type=str, default='replicate', choices=['replicate', 'constant', 'reflect', 'circular'])
parser.add_argument('--action_dim', nargs='+', type=int, default=(1, 1, 1, 1, 1, 1, 1), help='number of action dimention, xyz, rpy, gripper')
parser.add_argument('--action_activation', nargs='+', type=str, default=('tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh','sigmoid'), help='activation function for action')
parser.add_argument('--action_hidden_dim', type=int, default=128, help='hidden dimention of action')
parser.add_argument('--video_action_layers', type=int, default=12, help='number of action layers')

# data args
parser.add_argument("--sequence_length", type=int, default=6)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--src', type=str, default='/mnt/azureml/cr/j/bb7e6396fa4f48349d1ad48f61053a77/exe/wd/bridge2_processed/gpt4v-new')
parser.add_argument("--data_root", type=str, default="/mnt/robotdata/bridge2/images_bridge")
parser.add_argument('--dst_dir', type=str, default='../bridge2_processed/tokenized')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--num_shards', type=int, default=8)
parser.add_argument('--n_stacked_clips', type=int, default=10)

parser.add_argument('--weight_path', type=str, default='/mnt/data-rundong/VQ3D-vision-action/0515-action111-actionMask0.5/checkpoints/step_checkpoint-step_30000.ckpt')

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'

error_log = open(os.path.join(args.dst_dir, args.split, f'error_{args.gpu_id}.log'), 'a')

assert args.sequence_length == 6

dst_path = os.path.join(args.dst_dir, args.split, f'{args.gpu_id}.jsonl')
os.makedirs(os.path.dirname(dst_path), exist_ok=True)
dst_file = open(dst_path, 'w')

model = VQGANVisionAction(args)
state_dict = torch.load(args.weight_path, map_location='cpu')['state_dict']
model.load_state_dict(state_dict)
model = model.to(device).eval()

src_filepath = os.path.join(args.src, f'{args.split}.jsonl')

transform = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
])

with torch.no_grad():
    with open(src_filepath, 'r') as f:
        lines = f.readlines()
        num_data = len(lines)
        shard_size = num_data // args.num_shards + 1
        for line in tqdm(lines[args.gpu_id * shard_size: args.gpu_id * shard_size + shard_size]):

            instance_data = json.loads(line)
            instance_format = args.data_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}_' + str(instance_data['view']) + '.png'
            # each instance_info contains multiple frames, thus we need to handle them batch by batch
            num_frames = instance_data["frame_number"]

            num_start = num_frames // 6 + 1

            for start in range(0, num_start, args.n_stacked_clips):
                videos, actions = [], []
                for stack_cnt in range(args.n_stacked_clips):
                    if start + stack_cnt == num_start:
                        break
                    video, action = [], []
                    # if start is 0, encode 6 duplicate first frame and 6 null action
                    # else, encode frame 6i-6 to 6i-1 and action 6i-6 to 6i-1
                    if start+stack_cnt == 0: # video will be self.length duplicates of frame 0, and each action entry will be [0] * 7
                        img_filename = instance_format.format(instance_data['image_indices'][0])
                        img = Image.open(img_filename)
                        img = transform(img)
                        video = [img] * args.sequence_length
                        action = [[0. for _ in range(7)] for _ in range(args.sequence_length)]
                    else:
                        for i in range(6*(start+stack_cnt) - 6, 6*(start+stack_cnt)):
                            img_filename = instance_format.format(instance_data['image_indices'][i])
                            img = Image.open(img_filename)
                            img = transform(img)
                            video.append(img)
                            action.append(instance_data['actions'][i-1] if i > 0 else [0. for _ in range(7)])

                    videos.append(torch.stack(video).permute(1,0,2,3)) # [C, T, H, W])
                    actions.append(torch.tensor(action)) # [T, 7]

                videos = torch.stack(videos).to(device)
                actions = torch.stack(actions).to(device)
                n_stacked = videos.shape[0]

                recon_loss, recon_loss_action, _, _, vq_output, vq_output_action, _ = model(videos, actions)

                video_tokens, action_tokens = vq_output['encodings'].reshape(n_stacked, -1), vq_output_action['encodings'].reshape(n_stacked, -1)

                # add a line with the following entries: task_description (trajectory_language), scene_description (instance_data['descriptions'][0])
                # video_tokens, action_tokens, is_start_frame (0 or 1)
                try:
                    for stack_cnt in range(n_stacked):
                        ret = {
                            'trajectory_id': instance_data['trajectory_id'],
                            'view': instance_data['view'],
                            'start_frame': 6*(start+stack_cnt) - 6 if (start+stack_cnt) > 0 else -1,
                            'task_description': instance_data['trajectory_language'],
                            'scene_description': instance_data['descriptions']["0"],
                            'clip_description': instance_data['descriptions'][str(6*(start+stack_cnt)-1)] if (start+stack_cnt) != 0 else "",
                            'video_tokens': video_tokens.cpu().numpy().tolist(),
                            'action_tokens': action_tokens.cpu().numpy().tolist()
                        }
                        dst_file.write(json.dumps(ret) + '\n')
                        dst_file.flush()
                except:
                    error_log.write(line)
                    error_log.flush()

# randomly sample 90% of the data for training, and 10% for validation
# os.makedirs(os.path.join(args.dst_dir, 'train'), exist_ok=True)
# os.makedirs(os.path.join(args.dst_dir, 'val'), exist_ok=True)

# with open(dst_path, 'r') as f:
#     lines = f.readlines()
#     np.random.shuffle(lines)
#     num_train = int(len(lines) * 0.9)
#     train_lines = lines[:num_train]
#     val_lines = lines[num_train:]
#     with open(os.path.join(args.dst_dir, 'train', f'data_bridge2_processed_{args.gpu_id}.jsonl'), 'w') as f:
#         for line in train_lines:
#             f.write(line)
#     with open(os.path.join(args.dst_dir, 'val', f'data_bridge2_processed_{args.gpu_id}.jsonl'), 'w') as f:
#         for line in val_lines:
#             f.write(line)
