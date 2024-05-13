# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGANVisionAction, VideoData, get_image_action_dataloader, count_parameters
from tats.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from PIL import Image
# from tats.dataloader_img import get_image_dataloader


def main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()

    # trainer args
    parser.add_argument("--nodes", type=int, default=1, help="nodes")
    parser.add_argument("--devices", type=int, default=8, help="e.g., gpu number")
    parser.add_argument("--default_root_dir", type=str, default="logs/debug")
    parser.add_argument("--max_steps", type=int, default=300000, help="max_steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume from checkpoint")

    # model args
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_codes', type=int, default=2048)
    parser.add_argument('--n_hiddens', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--downsample', nargs='+', type=int, default=(2, 16, 16))
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
    parser.add_argument('--action_dim', type=int, default=7, help='number of action dimention, xyz, rpy, gripper')
    parser.add_argument('--action_hidden_dim', type=int, default=128, help='hidden dimention of action')
    parser.add_argument('--video_action_layers', type=int, default=3, help='number of action layers')

    # data args
    parser.add_argument(
        "--dataroot", type=str, default="/mnt/data-rundong/TATS/data_lists"
    )
    parser.add_argument("--sequence_length", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--val_check_interval', type=int, default=1.0)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_step_frequency', type=int, default=5000)

    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    try:
        print('MASTER_ADDR', os.environ['MASTER_ADDR'])
        print('MASTER_PORT', os.environ['MASTER_PORT'])
        print('NODE_RANK', os.environ['NODE_RANK'])
        print('LOCAL_RANK', os.environ['LOCAL_RANK'])
        print('RANK', os.environ['RANK'])
        print('WORLD_SIZE', os.environ['WORLD_SIZE'])
    except:
        pass

    # train_dataloader = get_image_dataloader(args, split='train')
    test_dataloader = get_image_action_dataloader(args, split='test')

    device = f'cuda:{args.gpu_id}'

    model = VQGANVisionAction(args).to(device)

    # load the most recent checkpoint file

    args.resume_from_checkpoint = '/mnt/data-rundong/VQGAN/temporal-down2-8nodes/checkpoints/step_checkpoint-step_80000.ckpt'
    
    assert args.resume_from_checkpoint is not None and os.path.exists(args.resume_from_checkpoint)
    ckpt = torch.load(args.resume_from_checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    for i, data in enumerate(test_dataloader):
        input_video = data['video'].to(device)
        recon_loss, x_recon, vq_output, perceptual_loss = model(input_video)
        # save the x_recon
        for j in range(x_recon.shape[0]):
            img = x_recon[j][:,0].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img + 0.5) * 255
            img = img.astype('uint8')
            img = Image.fromarray(img)
            img.save(f'logs/debug/recon_{i}_{j}.png')

            img_gt = input_video[j][:,0].detach().cpu().numpy().transpose(1, 2, 0)
            img_gt = (img_gt + 0.5) * 255
            img_gt = img_gt.astype('uint8')
            img_gt = Image.fromarray(img_gt)
            img_gt.save(f'logs/debug/gt_{i}_{j}.png')

        print(recon_loss, perceptual_loss)
        if i > 3:
            break

if __name__ == "__main__":
    main()
