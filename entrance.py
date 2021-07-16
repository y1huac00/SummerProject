import sys
import torch
from torchvision import models
from Calling_master import std_call_train
from Calling_master import single_train
from Calling_master import config
from Calling_master import test_model
"""
This is an entrance for receiving terminal command lines and 
calling training functions according to specified model and hyper-parameters.
"""

arg = sys.argv[1:]
l = ['model','target','n_class','phase','pretrained','batch_size','n_epochs','criterion','optimizer','learning_rate',
     'momentum','scheduler','step_size','gamma']

if len(arg) == 2:
    if arg[0] == 'tune' and arg[1] == 'default':
        print(str(config))
    elif arg[0] == 'tune' and arg[1] == 'm':
        print('make changes')

elif len(arg) == 13:
    modeldict = dict(zip(l,arg))
    input = input(str(modeldict)[1:-1].replace('\'','').replace(', ','\n') +
                  '\nContinue with the above model and hyper-parameters? ([y]/n): ')
    if input == 'y':
        model, optimizer, criterion, scheduler = [None,None,None,None]

        """------------------------------------load model------------------------------------"""
        if 'resnet' in modeldict['model']:
            if modeldict['model'] == 'resnet18':
                model = models.resnet18(pretrained=True if modeldict['pretrained'] == 't' else False)
            elif modeldict['model'] == 'resnet34':
                model = models.resnet34(pretrained=True if modeldict['pretrained'] == 't' else False)
            elif modeldict['model'] == 'resnet50':
                model = models.resnet50(pretrained=True if modeldict['pretrained'] == 't' else False)
            elif modeldict['model'] == 'resnet101':
                model = models.resnet101(pretrained=True if modeldict['pretrained'] == 't' else False)
            elif modeldict['model'] == 'resnet152':
                model = models.resnet152(pretrained=True if modeldict['pretrained'] == 't' else False)
            else:
                exit(modeldict['model'] + 'is not supported yet. Command \'python entrance.py help\' for information.')
            model.fc = torch.nn.Linear(model.fc.in_features,modeldict['n_class'])
        elif 'vgg' in modeldict['model']:
            if modeldict['model'] == 'vgg16':
                model = models.vgg16(pretrained=True if modeldict['pretrained'] == 't' else False)
                model.classifier[6] = torch.nn.Linear(4096,modeldict['n_class'])
            else:
                exit(modeldict['model'] + 'is not supported yet. Command \'python entrance.py help\' for information.')
        else:
            exit(modeldict['model'] + 'is not supported yet. Command \'python entrance.py help\' for information.')

        """-----------------------------------load optimizer------------------------------------"""
        if modeldict['optimizer'] == 'sgd':
            try:
                optimizer = torch.optim.SGD(params=model.parameters(), lr=modeldict['learning_rate'], momentum=modeldict['momentum'])
            except Exception as e:
                print(str(e))
                exit('Please check if learning_rate and momentum have been entered correctly. Command \'python entrance.py help\' for information.')
        else:
            exit(modeldict['optimizer'] + 'is not supported yet. Command \'python entrance.py help\' for information.')

        """------------------------------------load criterion------------------------------------"""
        if modeldict['criterion'] == 'cel':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            exit(modeldict['criterion'] + 'is not supported yet. Command \'python entrance.py help\' for information.')

        """------------------------------------load scheduler------------------------------------"""
        if modeldict['scheduler'] == 'steplr':
            try:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=modeldict['step_size'], gamma=modeldict['gamma'])
            except Exception as e:
                print(str(e))
                exit('Please check if step_size and gamma have been entered correctly. Command \'python entrance.py help\' for information.')
        else:
            exit(modeldict['scheduler'] + 'is not supported yet. Command \'python entrance.py help\' for information.')

        """------------------------------------train or test model------------------------------------"""
        if model['phase'] == 'train':
            model = single_train(model=model, target=modeldict['target'], batch_size=modeldict['batch_size'],
                                 n_epochs=modeldict['n_epochs'], criterion=criterion, optimizer=optimizer,
                                 scheduler=scheduler)
        elif model['phase'] == 'test':
            print('To be implemented')
    else:
        exit('Cancelled')
elif len(arg) == 1 and arg[0] == 'help':
    print('help')
else:
    exit('Use the command line \'python entrance.py help\' for information.')
