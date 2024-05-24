import os
import json
from tqdm import tqdm

root = './data_lists'

out_root = './data_lists_json'
os.makedirs(out_root, exist_ok=True)

json_filepath = '/mnt/robotdata/data_bridge2_processed.json'

splits = ['test', 'train']

for split in splits:
    scene_ids = []
    filename = os.path.join(root, split + '.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            scene_id = int(line.split('/')[-1].split('_')[1])
            if scene_id not in scene_ids:
                scene_ids.append(scene_id)

    # read json
    shard_id = 0
    out_filename_format = os.path.join(out_root, split, 'shard_{:03d}.jsonl')
    os.makedirs(os.path.join(out_root, split), exist_ok=True)
    with open(json_filepath, 'r') as f:
        lines = f.readlines()
        out_file = open(out_filename_format.format(shard_id), 'w')
        shard_data_cnt = 0
        print('Processing shard 0')
        for i, line in enumerate(tqdm(lines)):
            data = json.loads(line)
            if data['ID'] in scene_ids:
                out_file.write(line)
                shard_data_cnt += 1
            if shard_data_cnt == 1000:
                shard_id += 1
                shard_data_cnt = 0
                out_file.close()
                out_file = open(out_filename_format.format(shard_id), 'w')
                print('Processing shard {}'.format(shard_id))
