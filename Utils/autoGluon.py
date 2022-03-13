import autogluon.core as ag
from autogluon.vision import ImageDataset, ImagePredictor
import pandas as pd

roota = "../Plaindata/"
csv_file = '../species_auto.csv'
df = ImageDataset.from_csv(csv_file, root=roota)

train_data, test, _ = df.random_split(test_size=0.1)
print('train #:', len(train_data), 'test #:', len(test))

predictor = ImagePredictor()
model = ag.Categorical('resnet152_v2', 'resnest200', 'resnest269', 'se_resnet152_v1','se_resnet152_v2','vgg19',
                       'vgg19_bn','densenet201', 'inceptionv3', 'xception71', 'mobilenetv3_large', 'resnet152_v1e',
                       'senet_154e', 'darknet53', 'residualattentionnet236', 'cifar_residualattentionnet92',
                       'hrnet_w32_c', 'cait_xxs36_384','coat_tiny', 'convit_base', 'efficientnet_b8',
                       'gluon_senet154', 'vit_base_patch32_384', 'vit_base_r50_s16_224_in21k', 'vit_large_patch32_224_in21k',
                       "vit_small_resnet50d_s16_224", 'vit_tiny_r_s16_p8_384', 'vovnet39a', 'vovnet57a', 'wide_resnet101_2',
                       'xception', 'xception71')
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train_data, hyperparameters={'model': model,'epochs': 25},
              hyperparameter_tune_kwargs={'searcher': 'bayesopt', 'num_trials': 2})  # you can trust the default config, we reduce the # epoch to save some build time
fit_result = predictor.fit_summary()
print('Top-1 train acc: %.3f, val acc: %.3f' %(fit_result['train_acc'], fit_result['valid_acc']))