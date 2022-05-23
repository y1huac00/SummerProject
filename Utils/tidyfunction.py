import cv2
import numpy as np

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

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
    print('median:', arealist[round(len(arealist)/2)])

if __name__ == '__main__':
    # img = cv2.imread("D:/pythonproject/ostracod/test/HK14THL1C_104_105_50X.tif")
    img = cv2.imread("D:/pythonproject/ostracod/test/HK14THL1C_136_137_50X.tif")

    type = 'B'
    rang = (20000, 30000) if type == 'A' else (25000, 40000)

    resized, preprocessed_img = preprocess(img, type)
    findcontours(resized, preprocessed_img, lower=rang[0], upper=rang[1])

    cv2.imshow("Final Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO: straighten every grid and crop


