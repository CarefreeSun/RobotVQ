import os
import tqdm
import json

filepath = '/mnt/robotdata/RT1_all_data.json'

dst_filepath = './robot_datasets/tokenizer-training/rt1/rt1_formatted_wo_description.jsonl'

f = open(filepath, 'r')
os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
f_dst = open(dst_filepath, 'w')

lines = f.readlines()

for line in tqdm.tqdm(lines):
    data = json.loads(line)
    data_new = {}
    frame_numbers = data['Frame_number']

    for key in data.keys():
        if key == 'Visual':
            indices = [x for x in range(frame_numbers)]
            if frame_numbers % 6 != 0:
                data_new['image_indices'] = indices + [frame_numbers - 1] * (6 - frame_numbers % 6)
            else:
                data_new['image_indices'] = indices
        elif key == 'Action':
            data_new['actions'] = data['Action']
            if frame_numbers % 6 != 0:
                final_greeper_state = data['Action'][-1][-1] # 0 for closed, 1 for open
                data_new['actions'] += [[0.0] * 6 + [final_greeper_state]] * (6 - frame_numbers % 6)
                # data_new['actions'] += [data_new['actions'][-1]] * (6 - frame_numbers % 6)
        elif key == 'Frame_number':
            data_new['frame_number'] = (frame_numbers - 1) // 6 * 6 + 6
        elif key == 'ID':
            data_new['trajectory_id'] = data['ID']
        elif key == 'Text':
            data_new['task_description'] = data['Text']
        else:
            data_new[key] = data[key]

    assert 'image_indices' in data_new

    f_dst.write(json.dumps(data_new) + '\n')

f.close()
