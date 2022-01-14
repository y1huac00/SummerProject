import autogluon.core as ag
from autogluon.vision import ImageDataset, ImagePredictor
import pandas as pd

roota = "../Plaindata/.."
csv_file = '../species_auto.csv'
df = ImageDataset.from_csv(csv_file, root=roota)
print(df.head())

train_data, test, _ = df.random_split(test_size=0.1)
print('train #:', len(train_data), 'test #:', len(test))

predictor = ImagePredictor()
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train_data, hyperparameters={'epochs': 2})  # you can trust the default config, we reduce the # epoch to save some build time
fit_result = predictor.fit_summary()
print('Top-1 train acc: %.3f, val acc: %.3f' %(fit_result['train_acc'], fit_result['valid_acc']))