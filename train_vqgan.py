# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGANVision, VideoData, get_image_action_dataloader
from tats.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import math
import torch
# from tats.dataloader_img import get_image_dataloader


def main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()

    # trainer args
    parser.add_argument("--nodes", type=int, default=1, help="nodes")
    parser.add_argument("--devices", type=int, default=8, help="e.g., gpu number")
    parser.add_argument("--default_root_dir", type=str, default="logs/debug-2")
    parser.add_argument("--max_steps", type=int, default=100000, help="max_steps")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="init checkpoint")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume from checkpoint")

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

    # data args
    parser.add_argument("--split_root", type=str, default="/mnt/data-rundong/bridge2/gpt4v")
    parser.add_argument("--data_root", type=str, default="/mnt/robotdata/bridge2/images_bridge")
    parser.add_argument("--sequence_length", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--val_check_interval', type=int, default=1.0)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_step_frequency', type=int, default=5000)

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

    train_dataloader = get_image_action_dataloader(args, split='train', action=False)
    test_dataloader = get_image_action_dataloader(args, split='test', action=False)

    args.lr = args.lr * math.sqrt(args.nodes * args.devices * args.batch_size)

    model = VQGANVision(args)
    if args.load_checkpoint is not None:
        state_dict = torch.load(args.load_checkpoint, map_location='cpu')['state_dict']
        load_result = model.load_state_dict(state_dict, strict=True)
        # for missing_key in load_result.missing_keys:
        #     assert 'action' in missing_key.lower(), f"Missing key: {missing_key}"

    callbacks = []

    class StepCheckpointCallback(pl.Callback):
        def __init__(self, args):
            self.save_step_frequency = args.save_step_frequency
            self.log_interval = args.log_interval
            self.best_val_loss = float('inf')
            os.makedirs(args.default_root_dir, exist_ok=True)
            self.train_log = open(os.path.join(args.default_root_dir, "train_metrics.txt"), "w")
            self.eval_log = open(os.path.join(args.default_root_dir, "eval_metrics.txt"), "w")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if (trainer.global_step // 2) % self.save_step_frequency == 0:
                filepath = os.path.join(trainer.default_root_dir, 'checkpoints', f"step_checkpoint-step_{(trainer.global_step // 2)}.ckpt")
                trainer.save_checkpoint(filepath)
                # also save the latest checkpoint
                filepath = os.path.join(trainer.default_root_dir, 'checkpoints', f"latest_checkpoint.ckpt")
                trainer.save_checkpoint(filepath)
                # save all callback metrics into a file
            
            if (trainer.global_step // 2) % self.log_interval == 0:
                self.train_log.write(f"Training at step {trainer.global_step // 2}\n")
                for key, val in trainer.callback_metrics.items():
                    self.train_log.write(f"{key}: {val:.4f}\t")
                self.train_log.write("\n")
                self.train_log.flush()

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            self.train_log.write(f"Training at epoch {trainer.current_epoch}\n")
            for key, val in trainer.callback_metrics.items():
                self.train_log.write(f"{key}: {val:.4f}\t")
            self.train_log.write("\n")
            self.train_log.flush()
        
        def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            current_val_loss = trainer.callback_metrics['val/recon_loss']
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                filepath = os.path.join(trainer.default_root_dir, 'checkpoints', f"best_val_loss.ckpt")
                trainer.save_checkpoint(filepath)
            # save all callback metrics into a file
            self.eval_log.write(f"Validation at step {trainer.global_step // 2}\n")
            for key, val in trainer.callback_metrics.items():
                self.eval_log.write(f"{key}: {val:.4f}\t")
            self.eval_log.flush()

    callbacks.append(ImageLogger(batch_frequency=200, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=200, max_videos=4, clamp=True))
    callbacks.append(StepCheckpointCallback(args))

    # load the most recent checkpoint file
    # base_dir = os.path.join(args.default_root_dir, "lightning_logs")
    checkpoint_dir = os.path.join(args.default_root_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.exists(os.path.join(checkpoint_dir, "latest_checkpoint.ckpt")):
        print(f"Resume from checkpoint {os.path.join(checkpoint_dir, 'latest_checkpoint.ckpt')}")

    logger = TensorBoardLogger(save_dir=args.default_root_dir, name="logs")

    trainer = pl.Trainer(
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        default_root_dir=args.default_root_dir,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True), # training GAN with DDP
        devices=args.devices,
        num_nodes=args.nodes,
        logger=logger,
        log_every_n_steps=args.log_interval,
        precision='32',
        max_steps=args.max_steps,
        sync_batchnorm=True,
    )

    trainer.fit(model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=test_dataloader,
                ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
