for i in {0..7}
do
    python encode_image_bridge.py --gpu_id $i --split 'train' & 
done