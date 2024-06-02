for i in {0..7}
do
    python encode.py --gpu_id $i --split 'train' & 
done