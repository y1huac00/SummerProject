import os
import pandas as pd
import torch
import time
import copy
import numpy as np
from functools import partial
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

# config = {
#     "lr": tune.loguniform(1e-5, 1e-1),
#     "batch_size": tune.grid_search([16, 32, 64, 112]),
#     "step_size": tune.uniform(3,8),
#     "gamma":tune.grid_search([0.01,0.05,0.1,0.2,0.5]),
#     "momentum":tune.grid_search([0.5,0.6,0.7,0.8,0.9])
# }

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
PATH = './Models/'+str(time.time())+'.pth'

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    timelist=[]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    trainloss = []
    valloss = []
    for epoch in range(1,num_epochs+1):
        epochsince = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        cnt = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                cnt += 1
                if cnt % 100 == 0:
                    print('finished ', cnt, ' batches')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                trainloss.append(epoch_loss)
            else:
                valloss.append(epoch_loss)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val':
                # tune.report(loss=epoch_loss, accuracy=epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        if epoch > 3:
            if (trainloss[-3] < trainloss[-2] < trainloss[-1]) and (valloss[-3] < valloss[-2] < valloss[-1]):
                print('Train loss has increased over 3 epochs. Break.')
                break

        timelist.append(time.time()-epochsince)
        print(f'Time for epoch {epoch}: {(timelist[-1] // 60):.0f}m {(timelist[-1] % 60):.0f}s.')
        remainingtime = (num_epochs-epoch)*(sum(timelist)/len(timelist))
        print(f'Estimated remaining time: {(remainingtime // 60):.0f}m {(remainingtime % 60):.0f}s.')
        finishingtime = time.localtime(time.time()+remainingtime)
        print(f'Estimated finishing time: {finishingtime.tm_year}/{finishingtime.tm_mon}/{finishingtime.tm_mday} {finishingtime.tm_hour}:{finishingtime.tm_min}')



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), PATH)
    return model


# Load train, test and validation data by phase. Phase = train, val and test. Target = genus and species
def load_data(phase, target, d_transfroms, batch_size=16):
    data_path = './Metadata/' + target + '_' + phase + '.csv'
    src_path = './Plaindata'
    data_out = CustomImageDataset(data_path, src_path, d_transfroms)
    data_size = len(data_out)
    data_loader = DataLoader(data_out, batch_size=batch_size, shuffle=True)
    return data_loader, data_size

# def std_call(model):
#     optimizer_ft = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config['momentum'])
#     exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#     criterion = torch.nn.CrossEntropyLoss()
#     model_ft = train_model(model, criterion, optimizer_ft,
#                           exp_lr_scheduler, num_epochs=25)
#     return model_ft

def testmodel(model):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in dataloaders['train']:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _,predictions = torch.max(outputs,1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            print(f'complete test {n_samples}')
        acc = 100.0 * n_correct / n_samples
        print(f'accuracy = {acc}')

data_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                      transforms.CenterCrop([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])
target = 'species'
dataloaders = {}
dataset_sizes = {}
batch_size: int = 15
dataloaders['train'], dataset_sizes['train'] = load_data('train', target, data_transforms, batch_size)
dataloaders['val'], dataset_sizes['val'] = load_data('val', target, data_transforms, batch_size)
dataloaders['test'], dataset_sizes['test'] = load_data('test', target, data_transforms, batch_size)
dataiter = iter(dataloaders['train'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




train_features, train_labels = next(iter(dataloaders['train']))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
unloader = transforms.ToPILImage()
image = unloader(img)
plt.imshow(image)
print(device)

model_ft = models.resnet152(pretrained=True)

# model_ft.classifier[6] = torch.nn.Linear(4096,15)

num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 15)

model_ft = model_ft.to(device)

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
'''Need a parameter searching grid'''
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()
model_ft = train_model(model_ft, criterion, optimizer_ft,
                         exp_lr_scheduler, num_epochs=20)

#test
# model = models.resnet152(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 31)

# model = models.vgg16(pretrained=True)
# model.classifier[6] = torch.nn.Linear(4096,31)
# model.load_state_dict(torch.load('./Models/vgg16 1626284118.0475392.pth'))
# model.eval()
# model = model.to(device)
# testmodel(model)