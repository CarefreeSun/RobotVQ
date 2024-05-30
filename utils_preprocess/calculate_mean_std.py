import os
import json
import numpy as np
from tqdm import tqdm

def calculate_mean_std(data_dir, save_dir):
    '''
    calculate the mean and std of the actions values in the dataset
    '''
    data = np.empty((0, 7))
    f = open(data_dir, 'r')
    lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        instance_data = json.loads(line)
        actions =  instance_data['actions']
        action_values = np.array(actions)
        data = np.concatenate((data, action_values), axis=0)
    f.close()

    # calculate mean and std of each action
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print('mean:', mean)
    print('std:', std)
    save_path = os.path.join(save_dir, 'mean_std.json')
    with open(save_path, 'w') as f:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)

def incremental_mean_std(data_iterator, save_dir):
    n = np.zeros(7)
    mean = np.zeros(7)
    M2 = np.zeros(7)

    for data_chunk in tqdm(data_iterator):
        for x in data_chunk:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

    variance = M2 / (n - 1)
    std_dev = variance ** 0.5

    print(f"Mean: {mean}, Std Dev: {std_dev}")
    save_path = os.path.join(save_dir, 'mean_std.json')
    with open(save_path, 'w') as f:
        json.dump({'mean': mean.tolist(), 'std': std_dev.tolist()}, f)

def data_iterator(data_dir, chunk_size=200):
    f = open(data_dir, 'r')
    while True:
        action_data = np.empty((0, 7))
        for j in range(chunk_size):
            line = f.readline()
            if not line:
                break
            instance_data = json.loads(line)
            actions =  instance_data['actions']
            action_values = np.array(actions)
            action_data = np.concatenate((action_data, action_values), axis=0)
        if not line:
            break
        yield action_data

if __name__ == '__main__':
    data_dir = '../bridge2_processed/formatted.jsonl'
    save_dir = '../bridge2_processed'
    # calculate_mean_std(data_dir, save_dir)

    incremental_mean_std(data_iterator(data_dir), save_dir)
    # print(f"Mean: {mean}, Std Dev: {std_dev}")
            