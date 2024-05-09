import os
import argparse
from tats import VQGANVision, VQGANVisionAction
import torch
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_codes', type=int, default=16384)
    parser.add_argument('--n_hiddens', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--downsample', nargs='+', type=int, default=(2, 16, 16))
    parser.add_argument('--disc_channels', type=int, default=64)
    parser.add_argument('--disc_layers', type=int, default=3)
    parser.add_argument('--discriminator_iter_start', type=int, default=10000)
    parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
    parser.add_argument('--image_gan_weight', type=float, default=0.2)
    parser.add_argument('--video_gan_weight', type=float, default=0.2)
    parser.add_argument('--l1_weight', type=float, default=4.0)
    parser.add_argument('--gan_feat_weight', type=float, default=4.0)
    parser.add_argument('--perceptual_weight', type=float, default=4.0)
    parser.add_argument('--i3d_feat', action='store_true')
    parser.add_argument('--restart_thres', type=float, default=1.0)
    parser.add_argument('--no_random_restart', action='store_true')
    parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'group'])
    parser.add_argument('--padding_type', type=str, default='replicate', choices=['replicate', 'constant', 'reflect', 'circular'])
    parser.add_argument('--action_dim', type=int, default=7, help='number of action dimention, xyz, rpy, gripper')
    parser.add_argument('--action_hidden_dim', type=int, default=128, help='hidden dimention of action')
    parser.add_argument('--video_action_layers', type=int, default=3, help='number of action layers')
    parser.add_argument('--image_channels', type=int, default=3)

    # data and checkpoint args
    parser.add_argument("--load_checkpoint", type=str,
                default='/mnt/data-rundong/VQ3D-vision/temporal2-8node/checkpoints/step_checkpoint-step_20000.ckpt',
                help="init checkpoint")
    parser.add_argument("--src_filepath", type=str, 
                default='/mnt/data-rundong/bridge2/gpt4v/test.jsonl', 
                help="source file path")
    parser.add_argument("--dst_filepath", type=str, 
                # default='/mnt/data-rundong/bridge2/gpt4v-tokenized-vision/test.json', 
                default='/mnt/azureml/cr/j/bb7e6396fa4f48349d1ad48f61053a77/exe/wd/gpt4v-tokenized-vision/test.jsonl', 
                help="destination file path")
    parser.add_argument("--sequence_length", type=int, default=6)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--split_root", type=str, default="/mnt/data-rundong/bridge2/gpt4v")
    parser.add_argument("--data_root", type=str, default="/mnt/robotdata/bridge2/images_bridge")

    parser.add_argument('--tokenize_action', action='store_true')
    parser.add_argument("--resolution", type=int, default=256)

    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}'

    # prepare model
    if not args.tokenize_action:
        model = VQGANVision(args)
    else:
        model = VQGANVisionAction(args)

    state_dict = torch.load(args.load_checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)

    model = model.eval().to(device)

    lines = open(args.src_filepath, 'r').readlines()

    os.makedirs(os.path.dirname(args.dst_filepath), exist_ok=True)
    dst_file = open(args.dst_filepath, 'w')

    transform = transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
    ])

    for line in tqdm(lines[0:1]):
        data = json.loads(line)
        ret = {}
        # tokenize video
        video_tokens = []
        img_filepath_format = args.data_root + '/outputimage_' + str(data['trajectory_id']) + '_{}_' + str(data['view']) + '.png'
        for i in range(0, data['frame_number'] // 6 + 1):
            # if i == 0, tokenize the clip with six identical first frame
            # otherwise tokenize the clip from 6i-6 to 6i-1
            video = []
            if i == 0:
                img = Image.open(img_filepath_format.format(data['image_indices'][0]))
                img = transform(img)
                video = [img] * 6
            else:
                for j in range(6):
                    img = Image.open(img_filepath_format.format(data['image_indices'][6*i-6+j]))
                    img = transform(img)
                    video.append(img)
            video = torch.stack(video).permute(1, 0, 2, 3).to(device) # (C, T, H, W)
            with torch.no_grad():
                _, _, vq_output, *_ = model(video.unsqueeze(0))
                encoding_indices = vq_output['encodings'].flatten()

            video_tokens.append(encoding_indices.cpu().numpy().tolist()) # (frame_number//6+1, 3*256), 3 = 6 / 2

        # print(np.array(video_tokens).shape)

        if args.tokenize_action:
            # tokenize action
            actions = torch.tensor(data['actions']).to(device) # (frame_number, 7)
            with torch.no_grad():
                _, _, _, vq_output_action, *_ = model(video.unsqueeze(0), actions.unsqueeze(0))
                action_tokens = vq_output_action['encodings'].cpu().numpy().tolist()
    
        for key in data:
            if key not in ['image_indices', 'actions']:
                ret[key] = data[key]
            elif key == 'image_indices':
                ret['video_tokens'] = video_tokens
            else:
                if args.tokenize_action:
                    ret['action_tokens'] = action_tokens
                else:
                    ret['actions'] = data['actions']
        
        dst_file.write(json.dumps(ret) + '\n')

if __name__ == '__main__':
    main()


        