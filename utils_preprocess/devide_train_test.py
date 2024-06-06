import json
import random
import os
from tqdm import tqdm

filepath = '../robot_datasets/tokenizer-training/rt1/formatted_abs.jsonl'

train_split_path = '../robot_datasets/tokenizer-training/rt1/train.jsonl'
test_split_path = '../robot_datasets/tokenizer-training/rt1/test.jsonl'

os.makedirs(os.path.dirname(train_split_path), exist_ok=True)

f = open(filepath, 'r')
f_train = open(train_split_path, 'w')
f_test = open(test_split_path, 'w')

num_trajectory = 0

lines = f.readlines()
for line in lines:
    data = json.loads(line)
    num_trajectory = max(num_trajectory, data['trajectory_id'])

num_train = int(num_trajectory * 0.99)
ids = [x for x in range(num_trajectory)]
random.shuffle(ids)
train_ids = ids[:num_train]
test_ids = ids[num_train:]

for line in tqdm(lines):
    data = json.loads(line)
    if data['trajectory_id'] in train_ids:
        f_train.write(line)
    else:
        f_test.write(line)