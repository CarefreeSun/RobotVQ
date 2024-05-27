import os
import tqdm
import json

filepath = '/mnt/robotdata/bridge2/bridge2_gpt4v_annotated_merged_output_v2.jsonl'

dst_filepath = './bridge2_processed/formatted.jsonl'

f = open(filepath, 'r')
os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
f_dst = open(dst_filepath, 'w')

lines = f.readlines()

for line in tqdm.tqdm(lines):
    data = json.loads(line)
    data_new = {}
    frame_numbers = data['frame_number']

    if 'images' not in data:
        print(line)
        assert False

    for key in data.keys():
        if key == 'images':
            indices = [x for x in range(frame_numbers)]
            if frame_numbers % 6 != 0:
                data_new['image_indices'] = indices + [frame_numbers - 1] * (6 - frame_numbers % 6)
            else:
                data_new['image_indices'] = indices
        elif key == 'actions':
            data_new['actions'] = data['actions']
            if frame_numbers % 6 != 0:
                data_new['actions'] += [[0] * 7] * (6 - frame_numbers % 6)
                # data_new['actions'] += [data_new['actions'][-1]] * (6 - frame_numbers % 6)
        elif key == 'descriptions':
            data_new['descriptions'] = {}
            for key in data['descriptions']:
                # remove key if it is not a number
                if not key.isdigit():
                    print(line)
                    continue
                if int(key) != frame_numbers - 1:
                    data_new['descriptions'][key] = data['descriptions'][key]
                else:
                    data_new['descriptions'][(frame_numbers - 1) // 6 * 6 + 6 - 1] = data['descriptions'][key]
        elif key == 'frame_number':
            data_new['frame_number'] = (frame_numbers - 1) // 6 * 6 + 6
        else:
            data_new[key] = data[key]

    assert 'image_indices' in data_new

    f_dst.write(json.dumps(data_new) + '\n')

f.close()
