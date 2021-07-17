import sys
import torch
from torchvision import models
from Train import std_call_train
from Train import single_train
from Train import config
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
elif len(arg) == 14:
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
            model.fc = torch.nn.Linear(model.fc.in_features,int(modeldict['n_class']))
        elif 'vgg' in modeldict['model']:
            if modeldict['model'] == 'vgg16':
                model = models.vgg16(pretrained=True if modeldict['pretrained'] == 't' else False)
                model.classifier[6] = torch.nn.Linear(4096,int(modeldict['n_class']))
            else:
                exit(modeldict['model'] + 'is not supported yet. Command \'python entrance.py help\' for information.')
        else:
            exit(modeldict['model'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
        print(modeldict['model'] + ' has been loaded.')

        """-----------------------------------load optimizer------------------------------------"""
        if modeldict['optimizer'] == 'sgd':
            try:
                optimizer = torch.optim.SGD(params=model.parameters(), lr=float(modeldict['learning_rate']), momentum=float(modeldict['momentum']))
            except Exception as e:
                print(str(e))
                exit('Please check if learning_rate and momentum have been entered correctly. Command \'python entrance.py help\' for information.')
        else:
            exit(modeldict['optimizer'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
        print(modeldict['optimizer'] + ' has been loaded.')

        """------------------------------------load criterion------------------------------------"""
        if modeldict['criterion'] == 'cel':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            exit(modeldict['criterion'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
        print(modeldict['criterion'] + ' has been loaded.')

        """------------------------------------load scheduler------------------------------------"""
        if modeldict['scheduler'] == 'steplr':
            try:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(modeldict['step_size']),
                                                            gamma=float(modeldict['gamma']))
            except Exception as e:
                print(str(e))
                exit('Please check if step_size and gamma have been entered correctly. '
                     'Command \'python entrance.py help\' for information.')
        else:
            exit(modeldict['scheduler'] + ' is not supported yet. Command \'python entrance.py help\' for information.')
        print(modeldict['scheduler'] + ' has been loaded.')

        """------------------------------------train or test model------------------------------------"""
        if modeldict['phase'] == 'train':
            print('-----start training-----')
            try:
                model = single_train(model=model, target=modeldict['target'], batch_size=int(modeldict['batch_size']),
                                     n_epochs=int(modeldict['n_epochs']), criterion=criterion, optimizer=optimizer,
                                     scheduler=scheduler)
            except Exception as e:
                print(str(e))
                exit('Training failed.')
        elif model['phase'] == 'test':
            print('To be implemented')
    else:
        exit('Cancelled')
elif len(arg) == 1 and arg[0] == 'help':
    print('help')
else:
    exit('Use the command line \'python entrance.py help\' for information.')
