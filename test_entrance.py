import argparse
import sys
import torch
from torchvision import models
from Train import test_model, load_data

'''
    Default model in case no parameter of model provided.
    1. Get the model path
    2. Decode the model
    3. Doing classification and output the result
'''
MODEL_PATH = 'Models/acc_0.896_species_resnet50.pth'

def determine_model(arg_model):
    if arg_model.lower() == 'resnet18':
        model = models.resnet18()
    elif arg_model.lower() == 'resnet34':
        model = models.resnet34()
    elif arg_model.lower() == 'resnet101':
        model = models.resnet101()
    elif arg_model.lower() == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif arg_model.lower() == 'vgg16':
        model = models.vgg16()
    else:
        model = models.resnet50(pretrained=True)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    type=str,
                    default=MODEL_PATH,
                    help="Model path for your classification task.")
args = parser.parse_args()
model_info = args.model_path.split('_')[-1].split('.')[0]
model = determine_model(model_info)
target = args.model_path.split('_')[-2]
test_model(model, args.model_path, target)