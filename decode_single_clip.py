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
from tats import VQGANVisionActionEval, AverageMeter
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
parser.add_argument('--restart_thres', type=float, default=1.0)
parser.add_argument('--no_random_restart', action='store_true')
parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'group'])
parser.add_argument('--padding_type', type=str, default='replicate', choices=['replicate', 'constant', 'reflect', 'circular'])
parser.add_argument('--action_dim', nargs='+', type=int, default=(1, 1, 1, 1, 1, 1, 1), help='number of action dimention, xyz, rpy, gripper')
parser.add_argument('--action_activation', nargs='+', type=str, default=('none', 'none', 'none', 'none', 'none', 'none', 'sigmoid'), help='activation function for action')
parser.add_argument('--action_hidden_dim', type=int, default=128, help='hidden dimention of action')
parser.add_argument('--video_action_layers', type=int, default=12, help='number of action layers')

# data args
parser.add_argument("--normalize", action="store_true", help="normalize the actions")
parser.add_argument('--wo_transformer_residual', action='store_true', help='use transformer residual')
parser.add_argument("--sequence_length", type=int, default=6)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument("--filepath", type=str, default="../robot_datasets/0531-action111-bridge-noMask-woResidual/test_with_gt_action/8.jsonl")
parser.add_argument('--dst_dir', type=str, default='../eval_decoded/')
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--weight_path', type=str, default='/mnt/data-rundong/VQ3D-vision-action/0531-action111-bridge-noMask-woResidual/checkpoints/step_checkpoint-step_30000.ckpt')

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'

model = VQGANVisionActionEval(args)
state_dict = torch.load(args.weight_path, map_location='cpu')['state_dict']
result = model.load_state_dict(state_dict, strict=False)
for k in result.missing_keys:
    assert 'discriminator' in k or 'perceptual_model' in k
model = model.to(device).eval()

src_file = open(args.filepath, 'r')
lines = src_file.readlines()

transform = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
])

assert args.normalize and args.wo_transformer_residual
mean_std_path = os.path.join('../robot_datasets/tokenizer-training/bridge2', 'mean_std.json')
mean, std = json.load(open(mean_std_path, 'r'))['mean'], json.load(open(mean_std_path, 'r'))['std']
mean[-1] = 0.
std[-1] = 1.

action_dim_wise_normalized_meter = [AverageMeter() for _ in range(7)]

with torch.no_grad():
    for line in tqdm(lines[:20]):
        instance_data = json.loads(line)
        video_tokens = instance_data['video_tokens'] # 768 (3*256)
        action_tokens = instance_data['action_tokens']
        trajectory_id = instance_data['trajectory_id']
        view_id = instance_data['view']
        start_frame = instance_data['start_frame']
        action_gt = torch.tensor(instance_data['gt_actions']) # (6, 7)

        # decode video tokens
        video_tokens = torch.tensor(video_tokens, device=device).unsqueeze(0).reshape(1, 3, 16, 16)

        recon_frames = model.decode_video(video_tokens).squeeze(0).permute(1,0,2,3).detach().cpu() # 6, 3, 256, 256

        action_tokens = torch.tensor(action_tokens, device=device).unsqueeze(0).reshape(1, 6, 7)

        recon_actions = model.decode_action(action_tokens).squeeze(0).detach().cpu() # 6, 7
        if args.normalize:
            recon_actions = recon_actions * torch.tensor(std).unsqueeze(0) + torch.tensor(mean).unsqueeze(0)

        dst_dir = os.path.join(args.dst_dir, f'{trajectory_id}_{view_id}_{start_frame}')
        os.makedirs(dst_dir, exist_ok=True)
        # save the input images
        for i, frame in enumerate(recon_frames):
            img = (frame + 0.5).clamp(0,1).numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(dst_dir, f'{i}_input.png'))

        for i, meter in enumerate(action_dim_wise_normalized_meter):
            meter.update((recon_actions[:, i] - action_gt[:, i]).abs().mean().item())

        # save the input actions
        ret = {}
        ret['trajectory_id'] = trajectory_id
        ret['view_id'] = view_id
        ret['start_frame'] = start_frame
        ret['recon_actions'] = recon_actions.tolist()
        ret['gt_actions'] = action_gt.tolist()

        with open(os.path.join(dst_dir, 'preds.json'), 'w') as f:
            json.dump(ret, f)

print('action_dim_wise_normalized_meter')
for i, meter in enumerate(action_dim_wise_normalized_meter):
    print(f'{i}: {meter.avg}')

        




        


