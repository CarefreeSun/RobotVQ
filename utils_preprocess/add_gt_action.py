import os
import json

src_root = '/mnt/data-rundong/bridge2_processed/tokenized/test/'
formatted_file = './bridge2_processed/formatted.jsonl'

dst_root = './bridge2_processed/tokenized_with_gt_action/test/'
os.makedirs(dst_root, exist_ok=True)

# read the formatted file and store the data in a dictionary
formatted_data = {}
with open(formatted_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        instance_data = json.loads(line)
        trajectory_id = instance_data['trajectory_id']
        view = instance_data['view']
        action_data = instance_data['actions']
        formatted_data[(trajectory_id, view)] = action_data

num_shards = 8
for num_shard in range(num_shards):
    f_dst = open(os.path.join(dst_root, f'{num_shard}_stacked.jsonl'), 'w')
    file_path = os.path.join(src_root, f'{num_shard}_stacked.jsonl')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            instance_data = json.loads(line)
            trajectory_id = instance_data['trajectory_id']
            view = instance_data['view']
            start_frame = instance_data['start_frame']
            gt_actions = formatted_data[(trajectory_id, view)][start_frame+5:start_frame+11] if start_frame != -1 else [[0. for _ in range(7)]] + formatted_data[(trajectory_id, view)][0:5]
