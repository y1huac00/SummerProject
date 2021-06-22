import os
import numpy as np
import pandas as pd
import time
import torch
import csv

''' This is the script for saving model and classification result: '''
''' Functions to be implemented:
    1. A validation function to test accuracy on the test dataset
    2. A recording function to recording the classification result of original files. Outputs (filename,label)
    3. A visualization function to visualize the classification results
'''


# Return a dictionary mapping class string with number label
def get_class_meaning(target):
    guideline_path = target + '_guide.csv'
    guide_data = pd.read_csv(guideline_path, header=None)
    guide_index = guide_data.set_index(1).to_dict(orient='dict')
    mydict = guide_index[0]
    return mydict


# Classify image and write the results into
def verify_model(model, test_loader, device, target, data_size):
    since = time.time()
    outfile = './Results/' + str(int(since)) + target + '_result.csv'
    class_dict = get_class_meaning(target)
    with torch.no_grad():
        # iterate over batch
        for images, labels, paths in test_loader:
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
                    label_t = class_dict[label]
                    row.append(label_t)
                    pred_t = class_dict[prediction]
                    row.append(pred_t)
                    writer.writerow(row)
        f_guide.close
    accu = running_corrects.double() / data_size
    print('Current test Acc: {:4f}'.format(accu))
    return accu
