import cv2
import numpy as np

# img  = cv2.imread("D:/pythonproject/ostracod/test/HK14THL1C_104_105_50X.tif")
# # cv2.imshow("Image", img)
#
# scale_percent = 5
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# # cv2.imshow("gray", gray)
#
# blur = cv2.GaussianBlur(gray, (5,5), 2)
# cv2.imshow("blur", blur)
#
# # thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# ret, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh", thresh)
#
#
# blur = cv2.GaussianBlur(thresh, (5,5), 0)
# cv2.imshow("blur", blur)
#
# ret, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh", thresh)
# #
# # blur = cv2.GaussianBlur(thresh, (5,5), 0)
# # cv2.imshow("blur", blur)
# #
# # ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
# # cv2.imshow("thresh", thresh)
# #
# # blur = cv2.GaussianBlur(thresh, (5,5), 0)
# # cv2.imshow("blur", blur)
# #
# # ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
# # cv2.imshow("thresh", thresh)
#
#
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# max_area = 0
# c = 0
# for i in contours:
#         area = cv2.contourArea(i)
#         if area > 1000:
#                 if area > max_area:
#                     max_area = area
#                     best_cnt = i
#                     image = cv2.drawContours(resized, contours, c, (0, 255, 0), 3)
#         c+=1
#
# # mask = np.zeros((gray.shape),np.uint8)
# # cv2.drawContours(mask,[best_cnt],0,255,-1)
# # cv2.drawContours(mask,[best_cnt],0,0,2)
# # cv2.imshow("mask", mask)
# #
# # out = np.zeros_like(gray)
# # out[mask == 255] = gray[mask == 255]
# # cv2.imshow("New image", out)
# #
# # blur = cv2.GaussianBlur(out, (5,5), 0)
# # cv2.imshow("blur1", blur)
# #
# # thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# # cv2.imshow("thresh1", thresh)
# #
# # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# c = 0
# for i in contours:
#         area = cv2.contourArea(i)
#         if area > 1000/2:
#             cv2.drawContours(resized, contours, c, (0, 255, 0), 3)
#         c+=1
#
#
# cv2.imshow("Final Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# dst = np.float32(thresh)
#
# dst = cv2.cornerHarris(dst,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# dst[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',dst)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()


file = '/Users/chenyihua/desktop/pythonprojects/ostracod data/testdata/A/HK14THL1C_24_25_50X/HK14THL1C_24_25_50X_grid_3.tif'
# file = '/Users/chenyihua/desktop/pythonprojects/ostracod data/testdata/A/HK14THL1C_24_25_50X/HK14THL1C_24_25_50X_grid_1.tif'
image = cv2.imread(file)
original_image = image
cv2.imshow('original_image', original_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.inRange(image, 70, 240)
cv2.imshow('threshold', image)

# ret, image = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)


for i in range(0, 5):
    kernel = np.ones((1, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=4)
    image = cv2.dilate(image, kernel, iterations=4)
    kernel = np.transpose(kernel)
    image = cv2.erode(image, kernel, iterations=4)
    image = cv2.dilate(image, kernel, iterations=4)

    # kernel = np.ones((2, 2), np.uint8)
    # # image = cv2.erode(image, kernel, iterations=2)
    # image = cv2.dilate(image, kernel, iterations=1)

# kernel = np.ones((1, 10), np.uint8)
# image = cv2.erode(image, kernel, iterations=5)
# image = cv2.dilate(image, kernel, iterations=5)
# kernel = np.transpose(kernel)
# image = cv2.erode(image, kernel, iterations=5)
# image = cv2.dilate(image, kernel, iterations=5)

cv2.imshow('erode and dilate', image)
# thresh = cv2.erode(thresh, kernel, iterations=1)

contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursimg = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
rectangleimg = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
arealist = []

for c, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area >= 10000:
        arealist.append(area)
        cv2.drawContours(contoursimg, contours, c, (0, 0, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(rectangleimg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(original_image, (x - 10, y - 10), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('contour', contoursimg)
cv2.imshow('rectangle box', original_image)

print(arealist)
cv2.imshow('i', rectangleimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
