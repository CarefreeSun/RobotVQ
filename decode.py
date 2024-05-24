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

parser.add_argument("--filepath", type=str, default="../eval_logs.jsonl")
parser.add_argument('--dst_dir', type=str, default='../eval_decoded/')
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--weight_path', type=str, default='/mnt/data-rundong/VQ3D-vision-action/0515-action111-actionMask0.5/checkpoints/step_checkpoint-step_30000.ckpt')

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'

# assert args.sequence_length == 6

# dst_path = os.path.join(args.dst_dir, args.split, f'{args.gpu_id}.jsonl')
# os.makedirs(os.path.dirname(dst_path), exist_ok=True)
# dst_file = open(dst_path, 'w')
# error_log = open(os.path.join(args.dst_dir, args.split, f'error_{args.gpu_id}.log'), 'a')

model = VQGANVisionAction(args)
state_dict = torch.load(args.weight_path, map_location='cpu')['state_dict']
model.load_state_dict(state_dict)
model = model.to(device).eval()

src_file = open(args.filepath, 'r')
lines = src_file.readlines()

transform = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
])

with torch.no_grad():
    for line in tqdm(lines):
        instance_data = json.loads(line)
        input_video_tokens = instance_data['input_video_tokens'] # 768 (3*256)
        input_action_tokens = instance_data['input_action_tokens']
        output_video_tokens_pred = instance_data['output_video_tokens_pred'] # 42 (6*7)
        output_video_tokens_gt = instance_data['output_video_tokens_gt']
        output_action_tokens_pred = instance_data['output_action_tokens_pred']
        output_action_tokens_gt = instance_data['output_action_tokens_gt']
        trajectory_id = instance_data['trajectory_id']
        view_id = instance_data['view']

        # decode video tokens
        input_video_tokens = torch.tensor(input_video_tokens, device=device).unsqueeze(0).reshape(1, 3, 16, 16)
        output_video_tokens_pred = torch.tensor(output_video_tokens_pred, device=device).unsqueeze(0).reshape(1, 3, 16, 16)
        output_video_tokens_gt = torch.tensor(output_video_tokens_gt, device=device).unsqueeze(0).reshape(1, 3, 16, 16)

        input_frames = model.decode_video(input_video_tokens).squeeze(0).permute(1,0,2,3).detach().cpu() # 6, 3, 256, 256
        output_frames_pred = model.decode_video(output_video_tokens_pred).squeeze(0).permute(1,0,2,3).detach().cpu()
        output_frames_gt = model.decode_video(output_video_tokens_gt).squeeze(0).permute(1,0,2,3).detach().cpu()

        input_action_tokens = torch.tensor(input_action_tokens, device=device).unsqueeze(0).reshape(1, 6, 7)
        output_action_tokens_pred = torch.tensor(output_action_tokens_pred, device=device).unsqueeze(0).reshape(1, 6, 7)
        output_action_tokens_gt = torch.tensor(output_action_tokens_gt, device=device).unsqueeze(0).reshape(1, 6, 7)

        input_actions = model.decode_action(input_action_tokens).squeeze(0).detach().cpu() # 6, 7
        output_actions_pred = model.decode_action(output_action_tokens_pred).squeeze(0).detach().cpu()
        output_actions_gt = model.decode_action(output_action_tokens_gt).squeeze(0).detach().cpu()


        dst_dir = os.path.join(args.dst_dir, f'{trajectory_id}_{view_id}')
        os.makedirs(dst_dir, exist_ok=True)
        # save the input images
        for i, frame in enumerate(input_frames):
            img = (frame + 0.5).clamp(0,1).numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(dst_dir, f'{i}_input.png'))
        # save the output predcition images
        for i, frame in enumerate(output_frames_pred):
            img = (frame + 0.5).clamp(0,1).numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(dst_dir, f'{i}_output_pred.png'))
        # save the output images
        for i, frame in enumerate(output_frames_gt):
            img = (frame + 0.5).clamp(0,1).numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(dst_dir, f'{i}_output_gt.png'))
        # save the input actions
        ret = {}
        ret['input_actions'] = input_actions.tolist()
        ret['output_actions_gt'] = output_actions_gt.tolist()
        ret['output_actions_pred'] = output_actions_pred.tolist()
        ret['task_scene_description'] = instance_data['task_scene_description']
        ret['input_clip_description'] = instance_data['input_clip_description']
        ret['output_clip_description_pred'] = instance_data['output_clip_description_pred']
        ret['output_clip_description_gt'] = instance_data['output_clip_description_gt']

        with open(os.path.join(dst_dir, 'preds.json'), 'w') as f:
            json.dump(ret, f)


        




        


