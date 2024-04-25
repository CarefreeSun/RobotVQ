import os
import numpy as np
from tqdm import tqdm

'''
In the root directory contains files in the format: prefix_{scene_id}_{frame_id}_{view_id}.png
we have stored their names into 'filenames.txt' file
We need to split the dataset into train and test set, with scene_id as the unique identifier
'''

def devide_train_test(split_ratio=0.9):
    root = '/mnt/azureml/cr/j/d8c986f48e0042758991784fe953c9c3/exe/wd/'
    with open('/mnt/azureml/cr/j/d8c986f48e0042758991784fe953c9c3/exe/wd/data-rundong/taming-transformers/data_lists/train.txt', 'r') as f:
        files = f.readlines()
    files = [f.strip() for f in files]
    # sort all files using scene_id, if the scene_id is the same, sort by view_id, and then by frame_id
    def sort_key(x):
        _, scene_id, frame_id, view_id = x.split('/')[-1].split('.')[0].split('_')
        return (int(scene_id), int(view_id), int(frame_id))
    files = sorted(files, key=sort_key)
    
    filesets = {}
    for f in files:
        _, scene_id, _, _ = f.split('/')[-1].split('_')
        if scene_id not in filesets:
            filesets[scene_id] = []
        filesets[scene_id].append(f)
    
    scenes = list(filesets.keys())
    # shuffle the scenes
    scenes = np.random.permutation(scenes)
    # take 100 scenes for debug
    # scenes = scenes[:100]
    train_scenes = scenes[:int(len(scenes) * split_ratio)]
    test_scenes = scenes[int(len(scenes) * split_ratio):]

    os.makedirs(os.path.join(root, 'data_lists'), exist_ok=True)
    f_train = open(os.path.join(root, 'data_lists/train.txt'), 'w')
    f_test = open(os.path.join(root, 'data_lists/test.txt'), 'w')
    for file in tqdm(files):
        _, scene_id, _, _ = file.split('/')[-1].split('_')
        if scene_id in train_scenes:
            f_train.write(file + '\n')
        elif scene_id in test_scenes:
            f_test.write(file + '\n')
    f_train.close()
    f_test.close()

if __name__ == '__main__':
    devide_train_test()
