import os
import json
from tqdm import tqdm

src_root = '../robot_datasets/0531-action111-bridge-noMask-woResidual/train/'
formatted_file = '../robot_datasets/tokenizer-training/bridge2/train.jsonl'

dst_root = '../robot_datasets/0531-action111-bridge-noMask-woResidual/train_with_gt_action/'
pred_next = False
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

num_shards = 1
start_shard = 8
for num_shard in range(start_shard, start_shard+num_shards):
    f_dst = open(os.path.join(dst_root, f'{num_shard}.jsonl'), 'w')
    file_path = os.path.join(src_root, f'{num_shard}.jsonl')
    lines = open(file_path, 'r').readlines()
    for line in tqdm(lines):
        instance_data = json.loads(line)
        trajectory_id = instance_data['trajectory_id']
        view = instance_data['view']
        start_frame = instance_data['start_frame']
        if pred_next:
            gt_actions = formatted_data[(trajectory_id, view)][start_frame+5:start_frame+11] if start_frame != -1 else [[0. for _ in range(6)] + [formatted_data[(trajectory_id, view)][0][-1]]] + formatted_data[(trajectory_id, view)][0:5]
        else:
            if start_frame == -1:
                gt_actions = [[0. for _ in range(6)] + [formatted_data[(trajectory_id, view)][0][-1]]] * 6
            else:
                gt_actions = formatted_data[(trajectory_id, view)][start_frame-1:start_frame+5] if start_frame != 0 else [[0. for _ in range(6)] + [formatted_data[(trajectory_id, view)][0][-1]]] + formatted_data[(trajectory_id, view)][0:5]
        instance_data['gt_actions'] = gt_actions
        f_dst.write(json.dumps(instance_data) + '\n')
