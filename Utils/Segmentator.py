"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2
import numpy as np
import os


def read_and_down(file_path, down_factor=32):
    '''
    For read and down scale the images.
    :param file_path: the path for original images
    :param down_factor: the scale down factor for original image
    :return: src: original image; dwn: scaled image
    '''
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    width = int(src.shape[1] // down_factor)
    height = int(src.shape[0] // down_factor)
    dim = (width, height)
    dwn = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
    # dwn = cv2.GaussianBlur(dwn, (3,3), 0)
    return src, dwn


def axis_candidate(projection):
    # Assumed separation: 12*5
    threshold = projection.max() * 0.5
    axis_candidates = []
    for p in range(0, projection.size):
        if projection[p] <= threshold:
            axis_candidates.append(p)
    return axis_candidates


def clean_candidates(candidates, projections, if_x=1):
    axis_breaks = []
    if if_x == 1:
        axis_factor = 12
    else:
        axis_factor = 5
    break_length = projections.size / axis_factor
    for c in range(1, len(candidates)):
        if candidates[c] - candidates[c - 1] > break_length / 3:
            axis_breaks.append(candidates[c - 1])
            axis_breaks.append(candidates[c])
    return axis_breaks


def grid_crop(x_cand, y_cand, trgt, file_string, scale):
    m = len(x_cand)
    n = len(y_cand)
    sub_images = []
    print(m * n)
    for y in range(0, n - 1, 2):
        for x in range(0, m - 1, 2):
            sub_images.append(trgt[y_cand[y] * scale:y_cand[y + 1] * scale, x_cand[x] * scale:x_cand[x + 1] * scale])
    p = 1
    for img in sub_images:
        fs = file_string + str(p) + '.tif'
        cv2.imwrite(fs, img)
        p += 1


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def sep_image(img_file, img_folder, thr_value=160, scale=32):
    img_root = ''.join(img_file.split(".")[:-1])
    img, scaled = read_and_down(os.path.join(img_folder, img_file), scale)

    thr = cv2.threshold(scaled, thr_value, 255, cv2.THRESH_BINARY_INV)[1]
    h_proj = thr.sum(axis=1)
    w_proj = thr.sum(axis=0)
    cnd_x = clean_candidates(axis_candidate(w_proj), w_proj, 1)
    cnd_y = clean_candidates(axis_candidate(h_proj), h_proj, 0)

    root_folder = os.path.join(img_folder, img_root)
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)
    # else:
    #     # already cropped
    #     return
    file_string = os.path.join(root_folder, (img_root + '_grid_'))

    grid_crop(cnd_x, cnd_y, img, file_string, scale)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def resize(img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def preprocess(img, type):
    if type == 'A':
        resized = resize(img, 10)
        resized = rotate_image(resized, 2)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        cv2.imshow("blur", blur)

        blur = cv2.medianBlur(blur, 9)
        cv2.imshow('median', blur)

        blur = cv2.GaussianBlur(blur, (9, 9), 0)
        cv2.imshow("blur", blur)

        blur = cv2.medianBlur(blur, 9)
        cv2.imshow('median', blur)

        ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)

        return resized, thresh

    elif type == 'B':
        resized = resize(img, 20)
        resized = rotate_image(resized, 4)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imshow("blur", blur)

        blur = cv2.medianBlur(blur, 3)
        cv2.imshow('median', blur)

        # blur = cv2.GaussianBlur(blur, (9, 9), 0)
        # cv2.imshow("blur", blur)
        #
        # blur = cv2.medianBlur(blur, 9)
        # cv2.imshow('median', blur)

        ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)

        kernel = np.ones((1, 6), np.uint8)  # note this is a horizontal kernel
        kernel = np.transpose(kernel)
        print(kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        kernel = np.ones((1, 6), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        # thresh = cv2.erode(thresh, kernel, iterations=1)
        cv2.imshow("dilated", thresh)

        return resized, thresh


def findcontours(draw_img, preprocessed_img, lower, upper):
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    arealist = []
    for i in contours:
        area = cv2.contourArea(i)
        print(area)
        if lower < area < upper:
            arealist.append(area)
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)
        c += 1
    arealist = sorted(arealist)
    print('arealist:', arealist)
    print('length:', len(arealist))
    print('median:', arealist[round(len(arealist) / 2)])


def solutionB(file, type):  # single file for test
    img = cv2.imread(file)
    rang = (20000, 30000) if type == 'A' else (25000, 40000)

    resized, preprocessed_img = preprocess(img, type)
    findcontours(resized, preprocessed_img, lower=rang[0], upper=rang[1])

    cv2.imshow("Final Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO: straighten every grid and crop from original image
    # TODO: modify SolutionA to work with straightened image
    # TODO: save to files
    # TODO: SolutionB to every file in a given directory


if __name__ == '__main__':
    A = True
    if A:  # Solution A: select candidates
        img_folder = 'E:/HKU_Study/PhD/Lab_work/Keyence_Images'
        for file in files(img_folder):
            print(file)
            sep_image(file, img_folder, 160, 16)
    else:  # Solution B: grid contour
        sampleA = ('D:/pythonproject/ostracod/test/HK14THL1C_104_105_50X.tif', 'A')
        sampleB = ('D:/pythonproject/ostracod/test/HK14THL1C_136_137_50X.tif', 'B')
        solutionB(sampleB[0], sampleB[1])
