for i in $(seq 0 7)
do
    python encode.py --gpu_id $i --split 'test' --normalize --wo_transformer_residual \
    --num_shards 8 --start_shard 0 \
    --n_stacked_clip 1 \
    --weight_path '/mnt/data-rundong/VQ3D-vision-action/0715-dinov2-action111-bridge-noMask-woResidual/checkpoints/step_checkpoint-step_25000.ckpt' &
done

python -c "import time; time.sleep(14*24*60*60)"