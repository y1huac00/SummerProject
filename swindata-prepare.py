import shutil
import pandas as pd
import os

'''
    The following function swindataset prepares the dataset for training a swin-transformer for classification by
    existing metadata and plaindata.
    It creates a directory under the current working directory as follow:
        ./swindataset
        ├── train_map.txt
        ├── train.zip
        ├── val_map.txt
        ├── val.zip
        ├── test_map.txt
        └── test.zip
'''

def swindataset(path_metadata, path_plaindata, train, val, test, dst):
    dfs = {j:pd.read_csv(os.path.join(path_metadata,i),names=['filename','label'],header=None) for i,j in zip([train, val, test],['train', 'val', 'test'])}
    os.makedirs(dst, exist_ok=True)
    for key in dfs:
        for img in dfs[key]['filename'].tolist():
            os.makedirs(os.path.join(dst, key), exist_ok=True)
            shutil.copyfile(src=os.path.join(path_plaindata, img), dst=os.path.join(dst, f'{key}/', img))
            print(img)

        shutil.make_archive(os.path.join(dst,key), 'zip', dst, key)
        shutil.rmtree(os.path.join(dst,key))

        dfs[key].to_csv(os.path.join(dst,f'{key}_map.txt'), header=None, index=None, sep=' ')


if __name__ == '__main__':
    swindataset(path_metadata='./Metadata/',
                path_plaindata='./Plaindata/',
                train='species_train.csv',
                val='species_val.csv',
                test='species_test.csv',
                dst='./swindataset/')