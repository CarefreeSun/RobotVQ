import os
import json
from tqdm import tqdm
import numpy as np
import torch

import torch

def rel2abs_gripper_actions(actions: torch.Tensor):
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions
    (0 for closed, 1 for open). Assumes that the first relative gripper is not redundant
    (i.e. close when already closed).
    """
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1

    # -1 for closing, 1 for opening, 0 for no change
    thresholded_actions = torch.where(opening_mask, torch.tensor(1), torch.where(closing_mask, torch.tensor(-1), torch.tensor(0)))

    def scan_fn(carry, i):
        return carry if thresholded_actions[i] == 0 else thresholded_actions[i]

    # if no relative grasp, assumes open for whole trajectory
    start_idx = (thresholded_actions != 0).nonzero(as_tuple=True)[0][0].item() if (thresholded_actions != 0).any() else 0
    start = -1 * thresholded_actions[start_idx]
    start = 1 if start == 0 else start

    new_actions = torch.zeros_like(actions)
    new_actions[0] = start

    for i in range(1, len(actions)):
        new_actions[i] = scan_fn(new_actions[i-1], i)

    new_actions = new_actions.float() / 2 + 0.5
    return new_actions

src_filepath = '../robot_datasets/tokenizer-training/rt1/rt1_formatted_wo_description.jsonl'
dst_filepath = '../robot_datasets/tokenizer-training/rt1/rt1_formatted_wo_description_action_abs.jsonl'

f = open(src_filepath, 'r')
lines = f.readlines()

dst_f = open(dst_filepath, 'w')

for line in tqdm(lines):
    data = json.loads(line)
    data_new = {}

    for key in data.keys():
        if key == 'actions':
            actions = torch.from_numpy(np.array(data['actions']))
            actions[:, -1] = rel2abs_gripper_actions(actions[:, -1])
            data_new['actions'] = actions.tolist()
        else:
            data_new[key] = data[key]
    dst_f.write(json.dumps(data_new) + '\n')

f.close()
dst_f.close()
    
        