for i in {0..7}
do
    python encode.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual \
    --num_shards 96 --start_shard 0 \
    --n_stacked_clip 1 \
    --weight_path '/mnt/data-rundong/VQ3D-vision-action/0602-action111-bridgeRT-noMask-woResidual-continue0531/checkpoints/step_checkpoint-step_50000.ckpt' & 
done
for i in {0..7}
do
    python encode.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual \
    --num_shards 96 --start_shard 8 \
    --n_stacked_clip 1 \
    --weight_path '/mnt/data-rundong/VQ3D-vision-action/0602-action111-bridgeRT-noMask-woResidual-continue0531/checkpoints/step_checkpoint-step_50000.ckpt' & 
done
for i in {0..7}
do
    python encode.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual \
    --num_shards 96 --start_shard 16 \
    --n_stacked_clip 1 \
    --weight_path '/mnt/data-rundong/VQ3D-vision-action/0602-action111-bridgeRT-noMask-woResidual-continue0531/checkpoints/step_checkpoint-step_50000.ckpt' & 
done
for i in {0..7}
do
    python encode.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual \
    --num_shards 96 --start_shard 32 \
    --n_stacked_clip 1 \
    --weight_path '/mnt/data-rundong/VQ3D-vision-action/0602-action111-bridgeRT-noMask-woResidual-continue0531/checkpoints/step_checkpoint-step_50000.ckpt' & 
done