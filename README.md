# Automated Taxonomic Identification Framework 
This is a project developed for identifying species of Ostracods. The project is still under construction. Since the Ostracods are not avaliable currently, photos 
of forums will be used to test the efficiency of the models.

What to expect in next stage?
* Testing all popular models on the project
* Identify popular species in Hong Kong
* Implement swin transformer
* Integrate auto ML learning frameworks (now AutoGluon integrated)
* Create easy to use procedure

# Development plan

## Part A: Visualize

* The ```cm_test.py``` file contains an un-implemented visualize method to show the confusion 
matrix of classification results. The next step is to generalize the function in that file
to make it runnable for other classification results. The second goal is to make the output 
confusion matrix more visually acceptable.

* There is no visualize method for AutoGluon implementation and Yolo implementation.

* Individual image visualization is not available. We can add a new blank area under the image
and paste text showing the classification result on it.

* A visualization for ```classification_result.csv``` could be included.

## Part B: Evaluation

* No evaluation method included for YOLO implementation.
* No evaluation method for AutoGluon result.

## Part C: Usability
* The process for include data is hard for non-technical users. A click-to-run script
is required.

## Part D: Growth
* More auto-ml framework could be tested.
* More object detection method could be tested.
* Future direction of the project should be more focus on object detection.

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

#### To train a swin-transformer model, please refer to [how_to_train_swin.md](https://github.com/H-Jamieu/SummerProject/blob/master/how_to_train_swin.md)

By default, Resnet 50 is select for training classification. Run following script to start training:

```
python entrance.py
```
You can also add parameters to do custom training by calling ```entrance.py```. For example:
```
python entrance.py --model=resnet152 --lr=0.01
```
There are tons of parameters could be tuned. Please refer to the code for more details.
Trained models will be saved under ```project/models/*```.

## Inference on test data

to be done

## AutoGluon

The project also explored auto machinelearning to test its suitability to be included 
for future use. Currently, only training is completed. 

Before running, it may be slightly hard for you to use AutoGluon on Windows if GPU training
is necessary for you. Please follow following setup to make ensure the running smooth.

1. Pre-request:
    ```
    python version >=3.6
   Windows 11
   WSL2
   CUDA 11.X installed both on Windows host system and WSL
   CUDNN of corresponding CUDA version installed both on Windows host system and WSL
    ```
2. Install AutoGluon:
    ```
    pip3 install -U pip
    pip3 install -U setuptools wheel
    pip3 install mxnet-cu112
    pip3 install --pre autogluon
    ```

After finishing installation, go to ```project/utils/``` and run following script to do training.
```
python autoGluon.py
```
The default setting could achieve 88% top-1 accuracy at 20th epoch.

## Pre-trained model
Resnet 152 with 86.95% top 1 validation accuracy: https://1drv.ms/u/s!Avhb6zEgsVg1naFz-XupQWBVyNt2yQ?e=KwnKv7

|pretrained model|class|train accuracy|val accuracy|test accuracy|n_epochs|criterion|optimizer|learning rate|momentum|scheduler|step size|gamma|link|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|resnet152|species|99.99%|90.97%|91%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yihuac_connect_hku_hk/ETzF8jD3d6RDo_Cot215fuoBfT1JVAD3ZoUwchDhobvLTw?e=4R0tRa|
|vgg16|species|99.56%|88%|88%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://drive.google.com/file/d/1fntYLBz5-c94LbWyZ-Dv4hTSLA20EUdP/view?usp=sharing|
|resnet152|genus|99.98%|95.44%|95.7%|20|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.5|https://drive.google.com/file/d/1dFaZgYShIg8q9f4izycdp3bFM-un8H_n/view?usp=sharing|
|vgg16|genus|99.7%|93%|93%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://drive.google.com/file/d/1SIyBLgVx6VGCuhUhGOqcRJ0-nOmB7tYb/view?usp=sharing|

Swin-Transformer
|model|class|train:val:test|swin-config|n_epochs|accuracy@1|accuracy@5|link|
|---|---|---|---|---|---|---|---|
|Swin-Tiny|species|80:10:10|swin_tiny_patch4_window7_224|200|92.30%|99.037%|https://drive.google.com/file/d/1rK4cTDeI4bels_e-BmkBYsdjqSrjpJch/view?usp=sharing|
|Swin-Small|species|80:10:10|swin_small_patch4_window7_224|100|92.26%|99.037%|https://drive.google.com/file/d/1PVjnn62b1uBFrTwgTGcyj_ysMRG2JrnW/view?usp=sharing|
|Swin-Base|species|80:10:10|swin_base_patch4_window7_224|125|92.38%|99.037%|https://drive.google.com/file/d/1NlSr39V-FJ35RFc8kZN6X23lUl4ddHyM/view?usp=sharing|
|Swin-Large|species|80:10:10|swin_large_patch4_window7_224_22kto1k_finetune|200|90.17%|99.037%|https://drive.google.com/file/d/1MaudEAKlbLVrKIgo0pjgMBbfBPP_0f0f/view?usp=sharing|
                
