import autogluon.core as ag
from autogluon.vision import ImageDataset, ImagePredictor
import pandas as pd

root = "/Users/avangerfamilyhu/Desktop/Lab_work/SummerProject/Data/.."
all_data = ImageDataset.from_folder(root)

train, val, test = all_data.random_split(val_size=0.1, test_size=0.1)
print('train #:', len(train), 'test #:', len(test))

predictor = ImagePredictor()
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train, hyperparameters={'epochs': 2})  # you can trust the default config, we reduce the # epoch to save some build time
