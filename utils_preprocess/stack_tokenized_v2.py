import os
import argparse
import json
import copy
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='/mnt/data-rundong/robot_datasets/0715-dinov2-action111-bridge-noMask-woResidual_step25000_tokenized/rt1')
parser.add_argument('--dst', type=str, default='/mnt/data-rundong/robot_datasets/0715-dinov2-action111-bridge-noMask-woResidual_step25000_tokenized_stacked/rt1')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--start_shard', type=int, default=0)
parser.add_argument('--num_shards', type=int, default=240)

args = parser.parse_args()

src_path = os.path.join(args.src, args.split)
for i in range(args.start_shard, args.start_shard+args.num_shards):
    src_filepath = os.path.join(src_path, f'{i}.jsonl')
    f = open(src_filepath, 'r')
    dst_filepath = os.path.join(args.dst, args.split, f'{i}.jsonl')
    os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
    dst_file = open(dst_filepath, 'w')
    # read and store all lines
    lines = f.readlines()
    n_lines = len(lines)
    line_cnt = -1
    while True:
        line_cnt += 1
        line = lines[line_cnt]
        instance_data = json.loads(line)
        trajectory_id = instance_data['trajectory_id']
        view = instance_data['view']
        start_frame = instance_data['start_frame']

        # if 6 duplicate first frames, next clip is just next line (frame 0)
        if start_frame == -1: 
            new_line_cnt = line_cnt + 1
        # else, next clip is 6 frames after
        else:
            new_line_cnt = line_cnt + 6
            
        if new_line_cnt == n_lines:
            break
        new_line = lines[new_line_cnt]

        
        new_instance_data = json.loads(new_line)
        new_trajectory_id = new_instance_data['trajectory_id']
        new_view = new_instance_data['view']
        if not (new_trajectory_id == trajectory_id and new_view == view):
            continue
        '''
        create a new data that stack these two instances, with the following fields
        - trajectory_id: a integer that identifies the trajectory
        - view: a string that describes the view
        - start_frame: the start frame of the clip, -1 means it is 6 duplicate first frames
        - task_description: a string that describes the task, identical for clips with the same trajectory_id
        - scene_description: a string that describes the initial scene, identical for clips with the same trajectory_id and view
        - input_clip_description: a string that describes the frame difference in the input clip
        - output_clip_description: a string that describes the frame difference in the output clip
        - input_video_tokens: a 2D array of size 768 (256 * 3),
            256 * 3 is because each clip has 6 frames and downsamples by factor 2
        - output_video_tokens: a 2D array of size 768 (256 * 3),
        - input_action_tokens: a 2D array of size 42 (6 * 7),
        - output_action_tokens: a 2D array of size 42 (6 * 7),
        '''
        stacked_instance = {}
        stacked_instance['trajectory_id'] = trajectory_id
        stacked_instance['view'] = view
        stacked_instance['start_frame'] = instance_data['start_frame']
        stacked_instance['task_description'] = instance_data['task_description']
        stacked_instance['scene_description'] = instance_data['scene_description']
        stacked_instance['input_clip_description'] = instance_data['clip_description']
        stacked_instance['output_clip_description'] = new_instance_data['clip_description']
        stacked_instance['input_video_tokens'] = instance_data['video_tokens']
        stacked_instance['output_video_tokens'] = new_instance_data['video_tokens']
        stacked_instance['input_action_tokens'] = instance_data['action_tokens']
        stacked_instance['output_action_tokens'] = new_instance_data['action_tokens']
        dst_file.write(json.dumps(stacked_instance) + '\n')


