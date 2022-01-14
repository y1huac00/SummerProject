# Automated Taxonomic Identification Framework 
This is a project developed for identifying species of Ostracods. The project is still under construction. Since the Ostracods are not avaliable currently, photos 
of forums will be used to test the efficiency of the models.

What to expect in next stage?
* Testing all popular models on the project
* Identify popular species in Hong Kong

# Start running
Following is a guide for you to start running using the sample foram dataset.
The project is tested under following setup:
```
Python version >= 3.6
numpy==1.17.0
matplotlib==3.3.4
torchvision>=0.8.2
seaborn==0.11.1
pandas==1.1.5
ray==1.4.0
tqdm==4.61.2
torch==1.7.1
Pillow==8.3.1
scikit_learn==0.24.2
```
Please refer to ```requirements.txt``` for the list of dependencies.

## Getting sample Data
Please download sample training file from https://1drv.ms/u/s!Avhb6zEgsVg1naFaP_H6N2qMgh5TFQ?e=PwpQQX
Those are collected foram images collected from http://endlessforams.org/ credited to Hsiang AY et al (2018).

To start deploying the data, just name the file Data.zip and un-zip it directly, you will get all images under file structure like: 
```
project
|--readme.md
|--Data
|   |--Genus-species
|   |   |--data
|   |   |--iamges
|   |   |   |--img00001.jpg <- The actual image for classsification
```
If you are using the example dataset, after deploying the data, run ```image_handler.py``` to process the raw images and get the shots of forams. The processing time would take several minutes. Plain shots of forams are saved under ```project\Plaindata\*```.
Once got the plainshot, we could move to metadata creation stage.

## Metadata
The metadata files provided the information of images and its belonged classes. Partition of train and test data are performed based on 
metadata. Please refer to the following introductions.
<details>
    <summary>Introduction of provided metadata files</summary>
        <ul>
            <li><code>input.csv</code>: a file containing image names and species classes of forums from original data. It is the output from main.py.</li>
            <li><code>output.csv</code>: a file containing all information in the input.csv plus image sizes of data. It is the output from metadata_helper.py.</li>
            <li><code>species.csv</code>: a file containing image names and numbered species classes. It us the output from metadata_helper.py.<\li>
            <li><code>species_guide.csv</code>: a file mapping numbers to species. It is the output from metadata_helper.py. <\li>
            <li><code>genus.csv</code>: a file containing image names and numbered genus classes. It is the output from metadata_helper.py. <\li>
            <li><code>genus_guide.csv</code>: a file mapping numbers to genus. It is the output from metadata_helper.py. <\li>
            <li><code>genus_train.csv</code>: a file showing the train-test split results. It is the output from train_test_split.csv. <\li>
        </ul>
</details>

## Strat training model

By default, Resnet 152 is select for training classification. Run following script to start training:

```
python calling_master.py
```
You can also add parameters to do custom training by calling ```entrance.py```.
Trained models will be saved under ```project/models/*```.

## Inference on test data

to be done

## Pre-trained model
Resnet 152 with 86.95% top 1 validation accuracy: https://1drv.ms/u/s!Avhb6zEgsVg1naFz-XupQWBVyNt2yQ?e=KwnKv7

|pretrained model|class|train accuracy|val accuracy|test accuracy|n_epochs|criterion|optimizer|learning rate|momentum|scheduler|step size|gamma|link|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|resnet152|species|99.99%|90.97%|91%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yihuac_connect_hku_hk/ETzF8jD3d6RDo_Cot215fuoBfT1JVAD3ZoUwchDhobvLTw?e=4R0tRa|
|vgg16|species|99.56%|88%|88%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://drive.google.com/file/d/1fntYLBz5-c94LbWyZ-Dv4hTSLA20EUdP/view?usp=sharing|
|resnet152|genus|99.98%|95.44%|95.7%|20|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.5|https://drive.google.com/file/d/1dFaZgYShIg8q9f4izycdp3bFM-un8H_n/view?usp=sharing|
|vgg16|genus|99.7%|93%|93%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://drive.google.com/file/d/1SIyBLgVx6VGCuhUhGOqcRJ0-nOmB7tYb/view?usp=sharing|
