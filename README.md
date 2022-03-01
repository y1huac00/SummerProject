# Automated Taxonomic Identification Framework 
This is a project developed for identifying species of Ostracods. The project is still under construction. Since the Ostracods are not avaliable currently, photos 
of forums will be used to test the efficiency of the models.

What to expect in next stage?
* Testing all popular models on the project
* Identify popular species in Hong Kong

# Structure
There are some files to large and not suitable for uploading to git hub. This section will help you to building them
## Data
Please download sample training file from https://1drv.ms/u/s!Avhb6zEgsVg1naFaP_H6N2qMgh5TFQ?e=PwpQQX

Name the file Data.zip and un-zip it directly, you will get all images under structure like: Data/Specie_names/images/Image.jpg

## Metadata
* input.csv: a file containing image names and species classes of forums from orginal data. It is the output from main.py.
* output.csv: a file containing all infromation in the input.csv plus image sizes of data. It is the output from matadata_helper.py.
* species.csv: a file containing image names and numbered species classses. It us the output from metadata_helper.py.
* species_guide.csv: a file mapping numbers to species. It is the output from metadata_helper.py.
* genus.csv: a file containing image names and numbered genus classses. It is the output from metadata_helper.py.
* genus_guide.csv: a file mapping numbers to genus. It is the output from metadata_helper.py.
* genus_train.csv: a file showing the train-test split results. It is the output from train_test_split.csv

## Pre-trained model
Resnet 152 with 86.95% top 1 validation accurancy: https://1drv.ms/u/s!Avhb6zEgsVg1naFz-XupQWBVyNt2yQ?e=KwnKv7
<<<<<<< HEAD
|pretrained model|class|train accuracy|val accuracy|test acccuracy|n_epochs|criterion|optimizer|learning rate|momentum|scheduler|step size|gamma|link|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|resnet152|species|/|/|91%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yihuac_connect_hku_hk/ETzF8jD3d6RDo_Cot215fuoBfT1JVAD3ZoUwchDhobvLTw?e=4R0tRa|
|vgg16|species|/|88%|88%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://drive.google.com/file/d/1fntYLBz5-c94LbWyZ-Dv4hTSLA20EUdP/view?usp=sharing|
|resnet152|genus|99.98%|95.44%|95.7%|20|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.5|https://drive.google.com/file/d/1dFaZgYShIg8q9f4izycdp3bFM-un8H_n/view?usp=sharing|
|vgg16|genus|99.7%|93%|93%|25|CrossEntropyLoss|SGD|0.001|0.9|StepLR|5|0.1|https://drive.google.com/file/d/1SIyBLgVx6VGCuhUhGOqcRJ0-nOmB7tYb/view?usp=sharing|
=======
>>>>>>> parent of a8f9790 (add a table)
