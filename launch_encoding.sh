for i in {0..7}
do
    python encode.py --gpu_id $i --split 'train' --normalize --wo_transformer_residual --num_shards 16 & 
done