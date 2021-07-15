import os
import numpy as np
import pandas as pd
import time
import torch
import csv
from torchvision import transforms, models
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sn

PATH = './Models/0.8695_acc.pth'

''' This is the script for saving model and classification result: '''
''' Functions to be implemented:
    1. A validation function to test accuracy on the test dataset
    2. A recording function to recording the classification result of original files. Outputs (filename,label)
    3. A visualization function to visualize the classification results
'''


# Return a dictionary mapping class string with number label
def get_class_meaning(target):
    """
    Map number to actual classes
    :param target: Species or genus.
    :return: Dictionary. eg: {0, 'Genus_Specie'}
    """
    guideline_path = target + '_guide.csv'
    guide_data = pd.read_csv(guideline_path, header=None)
    guide_index = guide_data.set_index(1).to_dict(orient='dict')
    mydict = guide_index[0]
    return mydict


def extract_class_label(target):
    """
    Get a list of classes
    :param target: Species or genus.
    :return: List. List of genus or species.
    """
    guideline_path = target + '_guide.csv'
    guideline_data = pd.read_csv(guideline_path, header=None)
    all_classes = guideline_data[0].values
    return all_classes


# Classify image and write the results into
def verify_model(model, test_loader, device, target, data_size):
    """
    Get the classified label of each image. Return a classification results csv file in the Results Directory.
    :param model: A trained CNN model.
    :param test_loader: An instance of Data_loader class. Defines the source of image.
    :param device: Cpu or cuda.
    :param target: Species or genus.
    :param data_size: Size of the data, i.e. how many images.
    :return: Accuracy of the classification, float.
    """
    since = time.time()
    outfile = './Results/' + str(int(since)) + target + '_result.csv'
    class_dict = get_class_meaning(target)
    with torch.no_grad():
        # iterate over batch
        for images, labels, paths in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            running_corrects = 0
            running_corrects += torch.sum(predictions == labels.data)

            with open(outfile, 'a', encoding='ascii', errors='ignore') as f_guide:
                writer = csv.writer(f_guide)
                for label, prediction, path in zip(labels, predictions, paths):
                    row = [path]
                    label = int(label.numpy())
                    label_t = class_dict[label]
                    row.append(label_t)
                    prediction = int(prediction.numpy())
                    pred_t = class_dict[prediction]
                    row.append(pred_t)
                    writer.writerow(row)
        f_guide.close()
    accu = running_corrects.double() / data_size
    print('Current test Acc: {:4f}'.format(accu))
    return accu


def plot_prediction(test_file):
    """
    This function is visualizing confusion matrix of classification result.
    :param test_file: the file containing the classification result of tested model. It is expected to be a output from
    verify_model function with format of [image_path, predicted_label, real_label].
    :return: None
    """
    pred = pd.read_csv(test_file, header=None)
    y_pred, y_real = pred.iloc[:, 2], pred.iloc[:, 1]
    accCount = 0
    for y1, y2 in zip(y_pred, y_real):
        if y1 == y2:
            accCount += 1
    print(accCount / len(y_pred))
    classes = extract_class_label('species')
    cm = confusion_matrix(y_pred, y_real)
    df_cm = pd.DataFrame(cm, index=classes,
                         columns=classes)
    plt.figure(figsize=(14, 8))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def result_visualization():
    """
    Visualizing
    :return:
    """
    return 0


#plot_prediction('./Results/1625212258species_result.csv')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = models.resnet152(pretrained=True)
