for shard_cnt in $(seq 0 1) 
do
    start_shard=$((0 + $shard_cnt*8))  # Removed unnecessary variable and ensured correct arithmetic
    for i in $(seq 0 7)
    do
        python encode.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual \
        --num_shards 240 --start_shard $start_shard \
        --n_stacked_clip 1 \
        --weight_path '/mnt/data-rundong/VQ3D-vision-action/0715-dinov2-action111-bridge-noMask-woResidual/checkpoints/step_checkpoint-step_25000.ckpt' &
    done
done
python -c "import time; time.sleep(14*24*60*60)"