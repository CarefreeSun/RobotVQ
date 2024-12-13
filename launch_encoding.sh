for shard_cnt in $(seq 0 1) 
do
    start_shard=$(($1 + $shard_cnt*8))  # Removed unnecessary variable and ensured correct arithmetic
    for i in $(seq 0 7)
    do
        python encode_latest.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual \
        --num_shards 64 --start_shard $start_shard \
        --n_stacked_clips 1 \
        --weight_path '/mnt/data-rundong/RobotVQ/1210-lr1e-6-8node-8v100-bs4-seqlen3-attn12-7actdec4-drop01-crop-kla-dim1024-imagenetnorm-imagegan-ganfeatloss02/checkpoints/step_checkpoint-step_50000.ckpt' &
    done
done
python -c "import time; time.sleep(14*24*60*60)"