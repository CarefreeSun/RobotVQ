import os
import numpy as np
from tqdm import tqdm
import json

root = '/mnt/robotdata/data_bridge2_processed.json'

dst_root = './data_bridge2_json_shards'

# randomly split the json file into train and test set, and put into 10 shards

def split_train_test(split_ratio=0.95, n_shards=10):
    with open(root, 'r') as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    data = np.random.permutation(data)
    train_data = data[:int(len(data) * split_ratio)]
    test_data = data[int(len(data) * split_ratio):]
    os.makedirs(dst_root, exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'test'), exist_ok=True)
    for i in range(n_shards):
        with open(os.path.join(dst_root, 'train', f'train_{i}.json'), 'w') as f:
            f.write('\n'.join([json.dumps(d) for d in train_data[i::n_shards]]))
        with open(os.path.join(dst_root, 'test', f'test_{i}.json'), 'w') as f:
            f.write('\n'.join([json.dumps(d) for d in test_data[i::n_shards]]))

if __name__ == '__main__':
    split_train_test()
