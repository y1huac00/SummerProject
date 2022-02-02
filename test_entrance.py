import argparse
from Train import test_model
import torch
from torchvision import models

'''
    Default model in case no parameter of model provided.
    1. Get the model path
    2. Decode the model
    3. Doing classification and output the result
'''
MODEL_PATH = 'Models/1643726019_0.88_species_vgg16.pth'

def determine_model(arg_model, arg_pretrain, arg_classes):
    if arg_model.lower() == 'resnet18':
        model = models.resnet18(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'resnet34':
        model = models.resnet34(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'resnet101':
        model = models.resnet101(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'resnet152':
        model = models.resnet152(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    elif arg_model.lower() == 'vgg16':
        model = models.vgg16(pretrained=arg_pretrain)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=arg_classes)
    else:
        model = models.resnet50(pretrained=arg_pretrain)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, arg_classes)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    type=str,
                    default=MODEL_PATH,
                    help="Model path for your classification task.")
parser.add_argument("--pretrained",
                    type=bool,
                    default=False,
                    help="Indicator if you want the model to be pre-trained.")
parser.add_argument("--classes",
                    type=int,
                    default=31,
                    help="The count of classes for classification.")
args = parser.parse_args()
model_info = args.model_path.split('_')[-1].split('.')[0]
model = determine_model(model_info, args.pretrained, args.classes)
target = args.model_path.split('_')[-2]
test_model(model, args.model_path, target)