import ogbench
import numpy as np
import os

dataset_dir = 'ogbench/data'
# names = ['pointmaze-giant-navigate-v0']
names = ['pointmaze-teleport-navigate-v0']

# 1. download
ogbench.download_datasets(names, dataset_dir=dataset_dir)

# 2. unzip each .npz into a folder of .npy files
for name in names:
    for suffix in ['', '-val']:
        npz_path = os.path.join(dataset_dir, f'{name}{suffix}.npz')
        if not os.path.exists(npz_path):
            print(f'[skip] not found: {npz_path}')
            continue

        out_dir = os.path.join(dataset_dir, f'{name}{suffix}')
        os.makedirs(out_dir, exist_ok=True)

        data = np.load(npz_path)
        for key in data.files:
            np.save(os.path.join(out_dir, f'{key}.npy'), data[key])
            print(f'  {name}{suffix}: {key} {data[key].shape} {data[key].dtype}')
        print(f'[ok] extracted to {out_dir}/')