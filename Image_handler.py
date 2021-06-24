import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# A list to store possible fault images
fault = []

# Find boundary of images
def calculateBoundary(image, height, length):
    startingPXL = [height // 4, length // 2]
    top = -1
    bottom = height - 1
    left = -1
    right = length - 1
    low_bound = -1

    # Create a vertical line, find white area and upper/lower boundary of image
    for i in range(0, height):
        if top == -1 and int(image[i][startingPXL[1]][0]) + int(image[i][startingPXL[1]][1]) + int(
                image[i][startingPXL[1]][2]) > 6:
            top = i
        # Find white box
        if low_bound == -1 and i >= 30 and int(image[i][9][0]) + int(image[i][9][1]) + int(image[i][9][2]) >= 751:
            low_bound = i
        if top != -1 and int(image[i][startingPXL[1]][0]) + int(image[i][startingPXL[1]][1]) + int(
                image[i][startingPXL[1]][2]) <= 6:
            if i - top > 60:
                bottom = i
                break

    if low_bound != -1 and low_bound >= top*2:
        startingPXL[0] = low_bound//2

    for j in range(0, length):
        if left == -1 and int(image[startingPXL[0]][j][0]) + int(image[startingPXL[0]][j][1]) + int(
                image[startingPXL[0]][j][2]) > 6:
            left = j
            right = right-j
            break
        # if left != -1 and int(image[startingPXL[0]][j][0]) + int(image[startingPXL[0]][j][1]) + int(
        #         image[startingPXL[0]][j][2]) <= 6:
        #     if j - left > 60:
        #         right = j
        #         break
    # print('Height: ', height, ' Width: ', length)
    # print('Bottom: ', bottom, ' Bound: ', low_bound)
    # print('Top: ', top, 'Left: ', left)
    if bottom > low_bound != -1:
        bottom = low_bound
    if top == -1:
        top = 0
    if left == -1:
        left = 0
    # print('Bottom: ', bottom, ' Bound: ', low_bound)

    return top, bottom, left, right


def imageCorpping(im, name, trgDir):
    na = np.array(im)
    orig = na.copy()  # Save original

    length = na.shape[1]
    height = na.shape[0]
    top, bottom, left, right = calculateBoundary(na, height, length)

    if bottom == height - 1 or left == 0:
        error = name
        fault.append(error)
    ROI = orig[top:bottom, left:right]
    finalDir = os.path.join(trgDir, name)
    Image.fromarray(ROI).save(finalDir)


def load_images_from_folder(dir):
    images = []
    names = []
    for foldername in tqdm(os.listdir(dir)):
        if not foldername.startswith('.'):
            finalFolder = os.path.join(os.path.join(dir, foldername), 'images')
        for filename in os.listdir(finalFolder):
            img = Image.open(os.path.join(finalFolder, filename)).convert('RGB')
            if img is not None:
            #     images.append(img)
            #     names.append(filename)
                imageCorpping(img, filename, trgDir)
    return (images, names)


srcDir = './Data'
trgDir = './Plaindata'
print('Loading Files.')
image, name = load_images_from_folder(srcDir)
print('Loading Finished.')
cnt = 0
# for i, n in zip(image, name):
#     imageCorpping(i, n, trgDir)
#     cnt += 1
#     if cnt%1000 == 0:
#         print('Finished ', cnt, ' of 27730 images.')
print('Possible error images:')
print(fault)

# testIMG = './Image_test/730580_ex307653_obj00317.jpg'
# name = testIMG.split('/')[-1]
# im = Image.open(testIMG).convert('RGB')
# imageCorpping(im,name,'')
