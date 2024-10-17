# version 2, created on 2024.10.17
# every frame can be the start frame of an episode, not only 6x frames

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
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGANVisionActionEval, VQFinetuneEval
from tats.modules.callbacks import ImageLogger, VideoLogger
# from pytorch_lightning.strategies import DeepSpeedStrategy
from torchvision import transforms
from time import sleep

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
parser.add_argument('--action_activation', nargs='+', type=str, default=('none', 'none', 'none', 'none', 'none', 'none', 'sigmoid'), help='activation function for action')
parser.add_argument('--action_hidden_dim', type=int, default=128, help='hidden dimention of action')
parser.add_argument('--video_action_layers', type=int, default=12, help='number of action layers')

# data args
parser.add_argument("--sequence_length", type=int, default=6)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--src', type=str, default='/mnt/data-rundong/robot_datasets/tokenizer-training')
parser.add_argument("--dataset_names", nargs='+', type=str, 
                    default=("pizza_width", 
                            ))
parser.add_argument("--image_root", nargs='+', type=str, 
                    default=("/mnt/robotdata/datasets/pizza_robot", 
                            ))
parser.add_argument("--normalize", action="store_true", help="normalize the actions")
parser.add_argument('--dst_dir', type=str, default='')
parser.add_argument('--wo_transformer_residual', action='store_true', help='use transformer residual')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--num_shards', type=int, default=8)
parser.add_argument('--start_shard', type=int, default=0)
parser.add_argument('--n_stacked_clips', type=int, default=10)

parser.add_argument('--weight_path', type=str, default='/mnt/data-rundong/VQ3D-vision-action/0531-action111-bridge-noMask-woResidual/checkpoints/step_checkpoint-step_30000.ckpt')

def reset_gripper_width(x):
    return 0.0 if x > 0.07 else 1.0

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'

assert args.sequence_length == 6

assert args.normalize and args.wo_transformer_residual

args.dst_dir = '/mnt/data-rundong/robot_datasets/' + args.weight_path.split('/')[-3] + '_step10000' + '_tokenized'

model = VQFinetuneEval(args)
state_dict = torch.load(args.weight_path, map_location='cpu')['state_dict']
result = model.load_state_dict(state_dict, strict=False)
# for k in result.missing_keys:
#     assert 'discriminator' in k or 'perceptual_model' in k
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
])

with torch.no_grad():
    for (dataset_name, image_root) in zip(args.dataset_names, args.image_root):

        dst_path = os.path.join(args.dst_dir, dataset_name, args.split, f'{args.gpu_id+args.start_shard}.jsonl')
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        dst_file = open(dst_path, 'w')
        error_log = open(os.path.join(args.dst_dir, dataset_name, args.split, f'error_{args.gpu_id+args.start_shard}.log'), 'a')

        mean_std_path = os.path.join(args.src, dataset_name, 'mean_std.json')
        mean, std = json.load(open(mean_std_path, 'r'))['mean'], json.load(open(mean_std_path, 'r'))['std']
        mean[-1] = 0.
        std[-1] = 1.
        src_filepath = os.path.join(args.src, dataset_name, 'pizza_dataset_width.jsonl')
        with open(src_filepath, 'r') as f:
            lines = f.readlines()
            num_data = len(lines)
            shard_size = num_data // args.num_shards + 1
            for line in tqdm(lines[(args.gpu_id + args.start_shard) * shard_size: (args.gpu_id + args.start_shard) * shard_size + shard_size]):

                instance_data = json.loads(line)
                if dataset_name == 'bridge2':
                    instance_format = image_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}_' + str(instance_data['view']) + '.png'
                elif dataset_name == 'rt1':
                    instance_format = image_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}' + '.png'
                elif 'pizza' in dataset_name:
                    instance_format = image_root + '/' + str(instance_data['ID']) + '/' + str(instance_data['trajectory_id']) + '/images/right_rgb' + '/{:03d}' + '.jpg'
                else:
                    assert False
                # each instance_info contains multiple frames, thus we need to handle them batch by batch
                num_frames = instance_data["frame_number"]

                # 去掉最后补全用的重复帧
                prev_frame_id = -100
                for frame_pos in range(num_frames):
                    cur_frame_id = instance_data['image_indices'][frame_pos]
                    if cur_frame_id == prev_frame_id: # 重复
                        num_frames = frame_pos
                        break
                    # 未重复
                    prev_frame_id = cur_frame_id

                num_start = num_frames - 5

                for start in range(0, num_start, args.n_stacked_clips):
                    try:
                        videos, actions = [], []
                        for stack_cnt in range(args.n_stacked_clips): 
                            if start + stack_cnt == num_start:
                                break
                            video, action = [], []
                            # if start is 0, encode 6 duplicate first frame and 6 null action
                            # else, encode frame i - 1 to i + 4 and action i - 1 to i + 4
                            # note that frame id refers to the id in frame index list, not the actual frame id when collecting since some frame is lost

                            if start+stack_cnt == 0: # video will be self.length duplicates of frame 0, and each action entry will be [0] * 7
                                # img_filename = instance_format.format(instance_data['image_indices'][0])
                                if dataset_name == 'pizza':
                                    img_filename = instance_format.format(instance_data['image_indices'][0])
                                else:
                                    img_filename = instance_format.format(instance_data['image_indices'][0])
                                img = Image.open(img_filename)
                                img = transform(img)
                                video = [img] * args.sequence_length
                                action = [[0. for _ in range(6)] + [reset_gripper_width(instance_data['action_gripper'][0][-1])] for _ in range(args.sequence_length)]
                            else:
                                for i in range(start + stack_cnt - 1, start + stack_cnt + 5):
                                    img_filename = instance_format.format(instance_data['image_indices'][i])
                                    img = Image.open(img_filename)
                                    img = transform(img)
                                    video.append(img)
                                    action.append(instance_data['actions'][i][:-1] + [reset_gripper_width(instance_data['action_gripper'][i][-1])])

                            videos.append(torch.stack(video).permute(1,0,2,3)) # [C, T, H, W])
                            actions.append(torch.tensor(action)) # [T, 7]

                        videos = torch.stack(videos).to(device)
                        actions = torch.stack(actions).to(device)
                        if args.normalize:
                            actions = (actions - torch.tensor(mean).to(device)) / torch.tensor(std).to(device)
                        n_stacked = videos.shape[0]

                        _, _, vq_output, vq_output_action = model(videos, actions)

                        video_tokens, action_tokens = vq_output['encodings'].reshape(n_stacked, -1), vq_output_action['encodings'].reshape(n_stacked, -1)

                        for stack_cnt in range(n_stacked):
                            # search for proper clip description
                            disc_id = None
                            if (start + stack_cnt != 0):
                                start_frame = start + stack_cnt - 1
                                for i in range(max(0, start_frame - 1), min(num_frames, start_frame + 2)): # 搜索开始帧的前帧到后帧
                                    if str(instance_data['image_indices'][i]) in instance_data['descriptions']:
                                        disc_id = str(instance_data['image_indices'][i])

                            ret = {
                                'trajectory_id': instance_data['trajectory_id'],
                                'view': instance_data['view'],
                                'start_frame': instance_data['image_indices'][start + stack_cnt - 1] if (start+stack_cnt) > 0 else -1,
                                'task_description': instance_data['task_description'],
                                'scene_description': instance_data['scene_description'],
                                'clip_description': instance_data['descriptions'][disc_id] if (((start+stack_cnt) != 0) and (disc_id is not None)) else "",
                                'video_tokens': video_tokens[stack_cnt].tolist(),
                                'action_tokens': action_tokens[stack_cnt].tolist(),
                            }
                            dst_file.write(json.dumps(ret) + '\n')
                            dst_file.flush()
                    except:
                        error_log.write(line)
                        error_log.flush()