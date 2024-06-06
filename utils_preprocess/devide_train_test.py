import json
import random
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='bridge2')
args = parser.parse_args()

suffix = '_abs' if args.dataset_name == 'rt1' else ''

filepath = f'../robot_datasets/tokenizer-training/{args.dataset_name}/formatted{suffix}.jsonl'

train_split_path = f'../robot_datasets/tokenizer-training/{args.dataset_name}/train.jsonl'
test_split_path = f'../robot_datasets/tokenizer-training/{args.dataset_name}/test.jsonl'

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