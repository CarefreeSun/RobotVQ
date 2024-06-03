for i in {0..7}
do
    python encode.py --gpu_id $i --split 'test' --normalize --wo_transformer_residual --num_shards 16 --start_shard 8 & 
done