# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGANDeepSpeed, VideoData, get_image_video_dataloader
from tats.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
# from tats.dataloader_img import get_image_video_dataloader


def main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)

    # trainer args
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

    # parser = VQGAN.add_model_specific_args(parser)

    args = parser.parse_args()
    # args.default_root_dir = os.path.join(
    #     "/mnt/data-rundong/VQGAN", args.default_root_dir
    # )

    train_dataloader = get_image_video_dataloader(args, split='train')
    test_dataloader = get_image_video_dataloader(args, split='test')

    args.lr = args.lr * args.nodes * args.devices / 8.0 * args.batch_size / 4.0

    model = VQGANDeepSpeed(args)

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="val/recon_loss",
            save_top_k=3,
            mode="min",
            filename="latest_checkpoint",
        )
    )
    callbacks.append(
        ModelCheckpoint(
            every_n_train_steps=3000,
            save_top_k=-1,
            filename="{epoch}-{step}-{train/recon_loss:.2f}",
        )
    )
    callbacks.append(
        ModelCheckpoint(
            every_n_train_steps=10000,
            save_top_k=-1,
            filename="{epoch}-{step}-10000-{train/recon_loss:.2f}",
        )
    )
    callbacks.append(ImageLogger(batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))

    # load the most recent checkpoint file
    # base_dir = os.path.join(args.default_root_dir, "lightning_logs")
    # if os.path.exists(base_dir):
    #     log_folder = ckpt_file = ""
    #     version_id_used = step_used = 0
    #     for folder in os.listdir(base_dir):
    #         version_id = int(folder.split("_")[1])
    #         if version_id > version_id_used:
    #             version_id_used = version_id
    #             log_folder = folder
    #     if len(log_folder) > 0:
    #         ckpt_folder = os.path.join(base_dir, log_folder, "checkpoints")
    #         for fn in os.listdir(ckpt_folder):
    #             if fn == "latest_checkpoint.ckpt":
    #                 ckpt_file = "latest_checkpoint_prev.ckpt"
    #                 os.rename(
    #                     os.path.join(ckpt_folder, fn),
    #                     os.path.join(ckpt_folder, ckpt_file),
    #                 )
    #         if len(ckpt_file) > 0:
    #             args.resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_file)
    #             print(
    #                 "will start from the recent ckpt %s" % args.resume_from_checkpoint
    #             )

    # strategy = DeepSpeedStrategy(
    #     stage=2, offload_optimizer=True, cpu_checkpointing=True
    # )
    trainer = pl.Trainer(
        callbacks=callbacks,
        val_check_interval=1000,
        default_root_dir=args.default_root_dir,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        log_every_n_steps=10,
        strategy='ddp',
        # strategy=strategy,
        precision=16,
        max_steps=args.max_steps,
        sync_batchnorm=True,
    )

    trainer.fit(model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
