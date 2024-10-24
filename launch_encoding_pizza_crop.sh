for shard_cnt in $(seq 0 1) 
do
    start_shard=$(($1 + $shard_cnt*8))  # Removed unnecessary variable and ensured correct arithmetic
    for i in $(seq 0 7)
    do
        python encode_pizza_width_crop.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual \
        --num_shards 16 --start_shard $start_shard \
        --n_stacked_clip 1 \
        --weight_path '/mnt/data-rundong/VQ3D-vision-action/1023-ftpizza-crop-4ldec-dropout01-bs128-open0close1-lr1e-6-width07-continue-0725-step95000/checkpoints/step_checkpoint-step_10000.ckpt' &
    done
done
python -c "import time; time.sleep(1*8*60*60)"


# --weight_path '/mnt/data-rundong/VQ3D-vision-action/0920-finetune-pizza-lr1e-6-bs16-open0close1/checkpoints/step_checkpoint-step_56000.ckpt' &
