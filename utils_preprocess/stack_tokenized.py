import os
import argparse
import json
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='/mnt/data-rundong/robot_datasets/0531-action111-bridge-noMask-woResidual_tokenized/')
parser.add_argument('--dst', type=str, default='/mnt/data-rundong/robot_datasets/0531-action111-bridge-noMask-woResidual_tokenized_stacked/')
parser.add_argument('--split', type=str, default='train')
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
    # read the next line
    line = f.readline()
    instance_data = json.loads(line)
    while True:
        new_line = f.readline()
        if not new_line:
            break
        trajectory_id = instance_data['trajectory_id']
        view = instance_data['view']
        new_instance_data = json.loads(new_line)
        new_trajectory_id = new_instance_data['trajectory_id']
        new_view = new_instance_data['view']
        if not (new_trajectory_id == trajectory_id and new_view == view):
            instance_data = copy.deepcopy(new_instance_data)
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

        instance_data = copy.deepcopy(new_instance_data)

