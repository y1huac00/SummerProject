import os
import pandas as pd
import torch
import time
import copy
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


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

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
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
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Load train, test and validation data by phase. Phase = train, val and test. Target = genus and species
def load_data(phase, target, d_transfroms, batch_size=16):
    data_path = './Metadata/' + target + '_' + phase + '.csv'
    src_path = './Plaindata'
    data_out = CustomImageDataset(data_path, src_path, d_transfroms)
    data_size = len(data_out)
    data_loader = DataLoader(data_out, batch_size=batch_size, shuffle=True)
    return data_loader, data_size


data_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                      transforms.CenterCrop([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])
target = 'species'
dataloaders = {}
dataset_sizes = {}
batch_size: int = 64
dataloaders['train'], dataset_sizes['train'] = load_data('train', target, data_transforms, batch_size)
dataloaders['val'], dataset_sizes['val'] = load_data('val', target, data_transforms, batch_size)
# dataloaders['test'], dataset_sizes['test'] = load_data('test', target, data_transforms, batch_size)

print(dataset_sizes)

train_features, train_labels = next(iter(dataloaders['train']))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
unloader = transforms.ToPILImage()
image = unloader(img)
plt.imshow(image)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_ft = models.resnet152(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 31)

model_ft = model_ft.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

'''Need a parameter searching grid'''
optimizer_ft = torch.optim.ASGD(model_ft.parameters(), lr=0.001,lambd=0.0002)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model_conv = train_model(model_ft, criterion, optimizer_ft,
                         exp_lr_scheduler, num_epochs=25)
torch.save(model_conv.state_dict(), PATH)
