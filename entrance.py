import sys

"""
This is an entrance for receiving terminal command lines and 
calling training functions according to specified model and hyper-parameters.
"""

arg = sys.argv[1:]
l = ['model','class','batch_size','n_epochs','criterion','optimizer','learning_rate','momentum','scheduler','step_size','gamma']

if len(arg) == 11:
    model = dict(zip(l,arg))
    input = input(str(model)[1:-1].replace('\'','').replace(', ','\n') + '\nContinue with the above model and hyperparameters? ([y]/n): ')
    if input == 'y':
        print('ok')
    else:
        exit('Cancelled')
else:
    exit('Use the command line \'python entrance.py help\' for information.')
